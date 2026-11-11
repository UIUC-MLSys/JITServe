"""Replica-specific priority for multi-replica power-of-K placement (§4.3).

The paper creates K dummy copies of a request on K sampled replicas, each with
a replica-specific priority, and keeps the copy that gets scheduled. Because
replicas are separate processes, we resolve that competition at arrival: each
sampled replica reports the priority the request would have there (given the
replica's measured per-token time and the higher-priority work already queued
ahead of it), and the router places the request on the replica where that
priority is highest, i.e. where its dummy would have won.
"""
from typing import Any, Dict, Optional, Sequence

from jitserve.request_info import RequestPhaseWeight, RequestType


def _ratio(numer: float, denom: float) -> float:
    if denom <= 0:
        return 1.0
    return max(1e-6, min(1.0, numer / denom))


def estimate_replica_priority(scheduler,
                              prompt_len: int,
                              predicted_output_len: int,
                              request_type: int,
                              slo_constraint: Sequence[float],
                              deadline: Optional[float] = None,
                              **_: Any) -> Dict[str, float]:
    queue = (list(scheduler.running) + list(scheduler.waiting) +
             list(scheduler.swapped))
    policy = scheduler.policy
    if policy is None:
        return {"priority": 0.0, "queue_len": len(queue), "n_ahead": len(queue)}

    penalty = scheduler.scheduler_config.penalty_factor
    batch_size = max(scheduler.max_batch_size, 1)
    unit_time = policy._get_interval_time()
    output_len = max(int(predicted_output_len), 1)
    t_gen = output_len * unit_time
    ttft, tbt, e2el = slo_constraint
    request_type = RequestType(request_type)
    if request_type == RequestType.LATENCY:
        deadline = ttft + tbt * output_len
    elif deadline is None:
        deadline = e2el
    decode_weight = RequestPhaseWeight.DECODE.value

    def goodput(wait: float) -> float:
        if request_type == RequestType.LATENCY:
            prefill = prompt_len * _ratio(ttft, wait + unit_time)**penalty
            decode = (output_len * decode_weight *
                      _ratio(deadline, wait + t_gen)**penalty)
            return prefill + decode
        return ((prompt_len + output_len * decode_weight) *
                _ratio(deadline, wait + t_gen)**penalty)

    # Requests this one would have to wait behind on this replica.
    own = goodput(0.0) / t_gen
    new_key = policy.arrival_key(output_len, deadline)
    ahead = []
    for sg in queue:
        if sg.is_finished():
            continue
        # JITServe ranks by goodput density, baselines by their own sort key.
        if policy.ranks_by_density:
            is_ahead = -policy.get_priority(sg)[0] >= own
        else:
            is_ahead = policy.get_priority(sg) < new_key
        if is_ahead:
            ahead.append(unit_time * policy._remaining_len(sg))
    wait = 0.0
    if len(ahead) >= batch_size:
        ahead.sort()
        # Slots free up as the shortest of the higher-priority requests end.
        wait = ahead[len(ahead) - batch_size]
    return {
        "priority": goodput(wait) / t_gen,
        "queue_len": len(queue),
        "n_ahead": len(ahead),
        "wait": wait,
    }

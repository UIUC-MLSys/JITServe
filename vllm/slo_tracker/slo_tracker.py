import time

from typing import Iterable, Optional
from vllm.slo_tracker.request_info import RequestType, RequestPhaseWeight
from vllm.sequence import SequenceGroup


class SLOTracker:
    def __init__(self, penalty_factor: int, interval_time: Optional[float] = None):
        self.penalty_factor = penalty_factor
        self.interval_time = interval_time

    def update_seq_group_metrics(self, seq_groups: Iterable[SequenceGroup]) -> None:
        cur_time = time.time()
        for seq_group in seq_groups:
            self._update_metrics(seq_group, cur_time)

    def _update_metrics(self, seq_group: SequenceGroup, cur_time: float) -> None:
        metrics = seq_group.concord_metrics
        input_len = seq_group.first_seq.get_prompt_len()
        output_len = seq_group.first_seq.get_output_len()

        self._maybe_set_ttft(seq_group, metrics, cur_time)

        delta_input = input_len - metrics.prompt_len
        delta_output = output_len - metrics.output_len
        metrics.new_prefill_tokens = delta_input
        metrics.new_decode_tokens = delta_output
        metrics.service_gain += self._compute_service_gain(
            seq_group, cur_time, input_len, output_len, delta_input,
            delta_output)

        self._update_tbt(metrics, cur_time, delta_output)

        if delta_output > 0:
            metrics.last_schedule_time = cur_time

        metrics.prompt_len = input_len
        metrics.output_len = output_len

    def _maybe_set_ttft(self, seq_group: SequenceGroup, metrics,
                        cur_time: float) -> None:
        if seq_group.first_seq.get_output_len() < 1 or metrics.TTFT is not None:
            return
        metrics.TTFT = cur_time - metrics.arrival_time

    def _update_tbt(self, metrics, cur_time: float, delta_output: int) -> None:
        if metrics.last_schedule_time is None or delta_output <= 0:
            return
        delta_time = cur_time - metrics.last_schedule_time
        metrics.TBT.append(delta_time / delta_output)

    def _compute_service_gain(self, seq_group: SequenceGroup, cur_time: float,
                              input_len: int, output_len: int,
                              delta_input: int,
                              delta_output: int) -> float:
        if seq_group.request_type == RequestType.LATENCY:
            return self._latency_service_gain(seq_group, cur_time, input_len,
                                              output_len, delta_input,
                                              delta_output)
        if seq_group.request_type in (RequestType.THROUGHPUT,
                                      RequestType.COLLECTIVE):
            return self._throughput_service_gain(seq_group, delta_input,
                                                 delta_output, input_len,
                                                 output_len)
        return 0.0

    def _latency_service_gain(self, seq_group: SequenceGroup, cur_time: float,
                              input_len: int, output_len: int,
                              delta_input: int,
                              delta_output: int) -> float:
        metrics = seq_group.concord_metrics
        service = 0.0

        if metrics.service_gain == 0 and metrics.TTFT is not None:
            prefill_ratio = self._clamp_ratio(seq_group.TTFT_constraint,
                                              metrics.TTFT)
            service += prefill_ratio**self.penalty_factor * max(delta_input, 0)

        desire_decode_len = self._desired_decode_len(seq_group, cur_time)
        if delta_output > 0:
            if desire_decode_len <= 0:
                service += delta_output
            else:
                decode_ratio = self._clamp_ratio(output_len, desire_decode_len)
                service += decode_ratio**self.penalty_factor * delta_output * RequestPhaseWeight.DECODE.value

        return service

    def _throughput_service_gain(self, seq_group: SequenceGroup,
                                 delta_input: int, delta_output: int,
                                 input_len: int, output_len: int) -> float:
        metrics = seq_group.concord_metrics
        if delta_input > 0:
            return 0.0
        if delta_output <= 0 or metrics.TTLT is None:
            return 0.0

        if seq_group.request_type == RequestType.COLLECTIVE:
            deadline_penalty = 1.0
        else:
            deadline_penalty = min(
                1.0,
                (seq_group.deadline / metrics.TTLT)**self.penalty_factor)
        return input_len * deadline_penalty + output_len * deadline_penalty * RequestPhaseWeight.DECODE.value

    def _desired_decode_len(self, seq_group: SequenceGroup,
                            cur_time: float) -> float:
        metrics = seq_group.concord_metrics
        if metrics.TTFT is None:
            return 0.0
        req_ttft = min(metrics.TTFT, seq_group.TTFT_constraint)
        return (cur_time - req_ttft -
                seq_group.arrival_time) / seq_group.TBT_constraint

    @staticmethod
    def _clamp_ratio(numer: float, denom: float) -> float:
        if denom <= 0:
            return 1.0
        return max(1e-6, min(1.0, numer / denom))

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Deque, Optional, Callable, TYPE_CHECKING

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.logger import init_logger
from vllm.slo_tracker.request_info import RequestType
from vllm.sequence import SequenceGroup, SequenceStatus
from vllm.utils import Device
from vllm.core.scheduler import (ScheduledSequenceGroup, Scheduler,
                                 SchedulerOutputs, SchedulerPrefillOutputs,
                                 SchedulerRunningOutputs,
                                 SchedulerSwappedInOutputs,
                                 SchedulingBudget, PreemptionMode,
                                 scheduled_seq_group_builder)

if TYPE_CHECKING:
    from vllm.core.scheduler import Scheduler

logger = init_logger(__name__)
_SLO_POLICY_NAMES = {"concord", "jitserve", "slo"}


@dataclass
class SLOSchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """
    # Selected sequences that are going to be swapped in
    seq_group: ScheduledSequenceGroup
    is_prefill: bool
    is_infeasible: bool
    # The blocks to swap in.
    blocks_to_swap_in: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]

    @classmethod
    def create_empty(cls) -> "SLOSchedulerSwappedInOutputs":
        return SLOSchedulerSwappedInOutputs(
            seq_group=scheduled_seq_group_builder(),
            is_prefill=False,
            is_infeasible=False,
            blocks_to_swap_in=[],
            blocks_to_copy=[],
        )
        

@dataclass
class SLOSchedulerPrefillOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """
    # Selected sequences for prefill.
    seq_group: ScheduledSequenceGroup

    @classmethod
    def create_empty(cls) -> "SLOSchedulerPrefillOutputs":
        return SLOSchedulerPrefillOutputs(
            seq_group=scheduled_seq_group_builder(),
        )


class SLOSchedulerPreemptionOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """
    
    def __init__(
        self,
        prefill_result: SLOSchedulerPrefillOutputs,
        swapped_in_result: SLOSchedulerSwappedInOutputs,
        running_result: SchedulerRunningOutputs,
        force_preemption_count: int,
    ) -> None:
        self.prefill_result = prefill_result
        self.swapped_in_result = swapped_in_result
        self.running_result = running_result
        self.force_preemption_count = force_preemption_count

    @classmethod
    def create_empty(cls) -> "SLOSchedulerPreemptionOutputs":
        return SLOSchedulerPreemptionOutputs(
            prefill_result=SchedulerPrefillOutputs.create_empty(),
            swapped_in_result=SchedulerSwappedInOutputs.create_empty(),
            running_result=SchedulerRunningOutputs.create_empty(),
            force_preemption_count=0,
        )      


class SLOScheduler(Scheduler):
    def __init__(self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        super().__init__(scheduler_config, cache_config, lora_config,
                         pipeline_parallel_size, output_proc_callback)

        self._cur_rank = 0
        self._exec_order: List[int] = []
        self._bin_bounds: List[Tuple[int, int]] = []
        self._bins_by_index: List[List[SequenceGroup]] = []
        self._current_bin_queue: Deque[SequenceGroup] = deque()
        self._pending_swap_out_queue: Deque[SequenceGroup] = deque()
        self._current_bounds: Tuple[int, int] = (0, 0)
        self._current_is_last = False

        self.prev_no_swap_space = False

    def _schedule(self) -> "SchedulerOutputs":
        if self.policy.update_schedule_count() and self._test_enough_swap_space():
            result = self._schedule_slo()
        else:
            result = self._schedule_chunked_prefill()
        return result
    

    def _test_enough_swap_space(self) -> bool:
        num_cpu_blocks_number = self.block_manager.get_num_free_cpu_blocks()
        num_cpu_total_blocks_number = self.block_manager.num_total_cpu_blocks
        if self.prev_no_swap_space is True:
            if num_cpu_blocks_number / num_cpu_total_blocks_number > (
                    1 - self.block_manager.cpu_watermarks):
                self.prev_no_swap_space = False
                return True
            return False
        if num_cpu_blocks_number / num_cpu_total_blocks_number < self.block_manager.cpu_watermarks:
            self.prev_no_swap_space = True
            return False
        return True

    def _stop_dull_scheduling(self) -> List[SequenceGroup]:
        """Stop dull scheduling."""
        max_pending = getattr(self.scheduler_config, "dull_pending_limit",
                              100)
        max_idle_iters = getattr(self.scheduler_config, "dull_max_iters",
                                 2000)
        abort_seqs: List[SequenceGroup] = []
        pending_count = len(self.waiting) + len(self.swapped)
        if (len(self.running) == 0 and pending_count > 0
                and pending_count < max_pending):
            self.dull_iteration += 1
            if self.dull_iteration >= max_idle_iters:
                self.dull_iteration = 0
                # remove all infeasible sequence groups
                def _abort_queue(queue):
                    for seq_group in queue:
                        for seq in seq_group.get_seqs():
                            seq.status = SequenceStatus.FINISHED_ABORTED
                            self.free_seq(seq)
                        abort_seqs.append(seq_group)

                _abort_queue(self.waiting)
                self.waiting = deque()
                _abort_queue(self.swapped)
                self.swapped = deque()

        return abort_seqs

    def _schedule_slo_prefills(
        self,
        seq_group: SequenceGroup,
        num_new_tokens: int,
        num_lookahead_slots: int,
        enable_chunking: bool = False,
    ) -> "SLOSchedulerPrefillOutputs":
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert len(waiting_seqs) == 1, (
            "Waiting sequence group should have only one prompt "
            "sequence.")
        if not enable_chunking:
            num_prompt_tokens = waiting_seqs[0].get_len()
            assert num_new_tokens == num_prompt_tokens

        if num_new_tokens == 0:
            raise ValueError("Cannot schedule the sequence group.")

        # Can schedule this request.
        self._allocate_and_set_running(seq_group)

        if enable_chunking and self.scheduler_config.is_multi_step:
            blocks_to_copy: List[Tuple[int, int]] = []
            # init_multi_step_from_lookahead_slots happens in append_slots
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            # This assert will trip when a copy-on-write happens. This is
            # not a concern as the very first sequence-group block
            # allocation happens above. Still, we have the assert to
            # catch any edge-cases.
            assert not blocks_to_copy
        else:
            seq_group.init_multi_step_from_lookahead_slots(
                num_lookahead_slots,
                num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
                is_multi_step=self.scheduler_config.is_multi_step,
                enable_chunking=enable_chunking)

        # Queue requests that couldn't be scheduled.
        self.prev_prompt = True

        return SLOSchedulerPrefillOutputs(
            seq_group=ScheduledSequenceGroup(seq_group,
                                             token_chunk_size=num_new_tokens),
        )


    def _schedule_slo_swapped(
        self: "SLOScheduler",
        seq_group: SequenceGroup,
        num_new_tokens: int,
        num_lookahead_slots: int,
        enable_chunking: bool = False,
    ) -> "SLOSchedulerSwappedInOutputs":
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        # If the sequence group cannot be swapped in, stop.
        is_prefill = seq_group.is_prefill()

        self._swap_in(seq_group, blocks_to_swap_in)
        self._append_slots(seq_group, blocks_to_copy, enable_chunking)
        scheduled_seq_group = None

        if is_prefill:
            scheduled_seq_group = ScheduledSequenceGroup(seq_group,
                                                         token_chunk_size=num_new_tokens)
        else:
            scheduled_seq_group = ScheduledSequenceGroup(seq_group,
                                                         token_chunk_size=1)

        return SLOSchedulerSwappedInOutputs(
            seq_group=scheduled_seq_group,
            is_prefill=is_prefill,
            is_infeasible=False,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
        )

    def _seq_len(self, seq_group: SequenceGroup) -> int:
        try:
            return int(seq_group.first_seq.get_prompt_len() +
                       seq_group.first_seq.get_output_len())
        except Exception:
            return 0

    def _service_gain(self, seq_group: SequenceGroup) -> float:
        p = self.policy.get_priority(seq_group)
        try:
            if isinstance(p, (int, float)):
                return float(p)
            if isinstance(p, tuple):
                if isinstance(p[0], (int, float)):
                    return float(p[0])
                if isinstance(p[0], tuple) and isinstance(p[0][0],
                                                        (int, float)):
                    return float(p[0][0])
            return float(p)
        except Exception:
            return 1.0

    def _in_bounds(self, length: int, bounds, is_last_bin: bool) -> bool:
        start, end_open = bounds
        return (start <= length < end_open) or (is_last_bin and length >= start)

    def _compute_collection_demand(self, seqs: List[SequenceGroup],
                                   schedule_interval: float) -> float:
        if not seqs:
            return 0

        cur_time = time.time()
        decode_len = [seq.first_seq.get_decode_len() for seq in seqs]
        predict_output_len = [seq.request_info.output_len for seq in seqs]

        remaining_tokens = [
            predict_output_len[i] - decode_len[i] for i in range(len(seqs))
        ]
        deadlines = [seq.deadline + seq.arrival_time - cur_time for seq in seqs]

        total_remaining = sum(remaining_tokens)
        min_ttd = min(deadlines)

        if min_ttd <= 0:
            return 0

        required_bandwidth = total_remaining / min_ttd
        return required_bandwidth * schedule_interval

    def _recompute_bins(self) -> None:
        preemption_list = list(self.waiting) + list(self.swapped) + list(
            self.running)
        if not preemption_list:
            self._exec_order = []
            self._bin_bounds = []
            self._bins_by_index = []
            self._cur_rank = 0
            self._current_bin_queue.clear()
            self._pending_swap_out_queue.clear()
            return

        scored = [(self._service_gain(sg), sg) for sg in preemption_list]
        scored.sort(key=lambda x: x[0])
        k = self.top_k_selection
        top_mk = [sg for _, sg in scored[:self.scheduler_config.max_num_seqs *
                                        k]]

        top_mk.sort(key=self._seq_len)
        total_reqs_num = len(top_mk)
        mode = None
        if mode == "length":
            max_length = max(max(self._seq_len(sg) for sg in top_mk), 1024)
            bins = [[] for _ in range(k)]
            for sg in top_mk:
                bin_idx = min(k - 1,
                              math.floor(self._seq_len(sg) / (max_length / k)))
                bins[bin_idx].append(sg)
            bins = [b for b in bins if b]
        else:
            bins = []
            cur_bin = []
            for sg in top_mk:
                if len(cur_bin) < max(self.scheduler_config.max_num_seqs,
                                      total_reqs_num // k):
                    cur_bin.append(sg)
                else:
                    bins.append(cur_bin)
                    cur_bin = [sg]
            if cur_bin:
                bins.append(cur_bin)

        scores = []
        for i, blist in enumerate(bins):
            if not blist:
                scores.append((float("-inf"), i))
            else:
                pri = sum(self._service_gain(sg) for sg in blist)
                scores.append((pri, i))

        scores.sort(key=lambda x: x[0])
        self._exec_order = [i for pri, i in scores if pri != float("-inf")]
        self._cur_rank = 0
        self._bins_by_index = bins
        self._bin_bounds = [(min(self._seq_len(sg) for sg in b),
                             max(self._seq_len(sg) for sg in b) + 1)
                            for b in bins]

        if self._exec_order:
            self._switch_to_bin(self._cur_rank, initialize=True)

    def _switch_to_bin(self, rank: int, initialize: bool = False) -> None:
        if not self._exec_order:
            self._current_bin_queue.clear()
            self._pending_swap_out_queue.clear()
            return
        rank = rank % len(self._exec_order)
        exec_idx = self._exec_order[rank]
        bounds = self._bin_bounds[exec_idx]
        bin_number = len(self._bins_by_index)

        cur = deque(self._bins_by_index[exec_idx])
        seen = set(id(x) for x in cur)

        left, right = exec_idx - 1, exec_idx + 1
        new_lower, new_upper = bounds

        while len(cur) < self.scheduler_config.max_num_seqs and (
                left >= 0 or right < bin_number):
            candidates = []
            if left >= 0:
                candidates.append((abs(left - exec_idx), left))
            if right < bin_number:
                candidates.append((abs(right - exec_idx), right))
            candidates.sort()
            _, chosen_idx = candidates[0]

            if chosen_idx < exec_idx:
                left -= 1
            else:
                right += 1

            if len(self._bins_by_index[chosen_idx]) == 0:
                continue

            for sg in self._bins_by_index[chosen_idx]:
                if id(sg) not in seen:
                    cur.append(sg)
                    seen.add(id(sg))

            new_lower = min(new_lower, self._bin_bounds[chosen_idx][0])
            new_upper = max(new_upper, self._bin_bounds[chosen_idx][1])

        self._current_bounds = (new_lower, new_upper)
        self._current_is_last = (new_upper == self._bin_bounds[-1][1])
        self._current_bin_queue = cur

        pend = deque()
        for sg in list(self.running):
            if not self._in_bounds(self._seq_len(sg), self._current_bounds,
                                   self._current_is_last):
                pend.append(sg)
            elif id(sg) not in seen:
                cur.append(sg)
                seen.add(id(sg))
        self._pending_swap_out_queue = pend

    def _recompute_bins_sliding_window(self) -> None:
        preemption_list = list(self.waiting) + list(self.swapped) + list(
            self.running)
        if not preemption_list:
            self._exec_order = []
            self._bin_bounds = []
            self._bins_by_index = []
            self._cur_rank = 0
            self._current_bin_queue.clear()
            self._pending_swap_out_queue.clear()
            return

        k = self.scheduler_config.top_k_selection
        scored = [(self._service_gain(sg), sg) for sg in preemption_list]
        scored.sort(key=lambda x: x[0])
        top_mk = [sg for _, sg in scored[:self.max_batch_size * k]]
        top_mk.sort(key=self._seq_len)

        best_sum, best_idx = float("-inf"), 0
        batch_size = self.max_batch_size

        if len(top_mk) >= batch_size:
            cur_sum = sum(self._service_gain(sg) for sg in top_mk[:batch_size])
            best_sum, best_idx = cur_sum, 0

            for i in range(batch_size, len(top_mk)):
                cur_sum += self._service_gain(top_mk[i])
                cur_sum -= self._service_gain(top_mk[i - batch_size])
                if cur_sum < best_sum:
                    best_sum, best_idx = cur_sum, i - batch_size + 1

            best_window = top_mk[best_idx:best_idx + batch_size]
        else:
            best_window = top_mk

        collections = {}
        schedule_time = self.policy.schedule_interval * self.policy.interval_time \
                        if self.policy.interval_time else self.policy.schedule_interval * 0.02
        for seq in best_window:
            if seq.request_type == RequestType.COLLECTIVE:
                collections.get(seq.collection_id, []).append(seq)

        for cid, seqs in collections.items():
            demand = self._compute_collection_demand(
                self.policy.seq_group_dict[cid], schedule_time)
            provided = len(seqs) * self.policy.schedule_interval

            if provided < demand:
                candidates = self.policy.seq_group_dict[cid]
                candidates.sort(key=self._service_gain, reverse=True)

                for r in candidates:
                    if r not in best_window:
                        best_window.append(r)
                        provided += self.policy.schedule_interval
                        if provided >= demand or len(best_window) >= batch_size:
                            break

        self._current_bin_queue = deque(best_window)
        seen = {id(sg) for sg in best_window}

        if best_window:
            self._current_bounds = (min(self._seq_len(sg)
                                        for sg in best_window),
                                    max(self._seq_len(sg)
                                        for sg in best_window) + 1)
        else:
            self._current_bounds = (0, 0)
        self._current_is_last = True

        pend = deque()
        for sg in list(self.running):
            if not self._in_bounds(self._seq_len(sg), self._current_bounds,
                                   self._current_is_last):
                pend.append(sg)
            elif id(sg) not in seen:
                self._current_bin_queue.append(sg)
                seen.add(id(sg))
        self._pending_swap_out_queue = pend

    def _init_preemption_outputs(self, enable_chunking: bool,
                                 SchedulerPrefillOutputs,
                                 SchedulerSwappedInOutputs):
        prefill_result = SchedulerPrefillOutputs.create_empty()
        swapped_in_result = SchedulerSwappedInOutputs.create_empty()

        running_result = self._scheduler_running_outputs_cache[
            self.cache_id].get_object()
        running_result.blocks_to_swap_out.clear()
        running_result.blocks_to_copy.clear()
        running_result.decode_seq_groups.clear()
        running_result.prefill_seq_groups.clear()
        running_result.leftover_running.clear()
        running_result.preempted.clear()
        running_result.swapped_out.clear()
        running_result.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking)
        running_result.decode_seq_groups_list.clear()
        running_result.prefill_seq_groups_list.clear()

        prefill_result.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=True, enable_chunking=enable_chunking)
        swapped_in_result.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking)

        return prefill_result, swapped_in_result, running_result

    # Inject sequences from waiting and swapped queues within length bounds into the current bin queue.
    def _inject_current_bin(self) -> None:
        bounds = self._current_bounds
        is_last_bin = self._current_is_last

        seen = set(id(sg) for sg in self._current_bin_queue)
        for sg in list(self.waiting) + list(self.swapped):
            if id(sg) in seen:
                continue
            if self._in_bounds(self._seq_len(sg), bounds, is_last_bin):
                self._current_bin_queue.appendleft(sg)
                seen.add(id(sg))

    def _preempt_pending(self, running_result):
        swaps_budget = max(0, int(self.max_swaps_per_iter))
        force_preemption_count = 0
        swapped_this_iter = 0
        while swaps_budget > 0 and self._pending_swap_out_queue:
            sg = self._pending_swap_out_queue.popleft()
            if sg not in self.running or sg.first_seq.is_finished():
                continue
            if self.search_strategy in ["length_bin", None]:
                preempt_mode = self._preempt(
                    sg, running_result.blocks_to_swap_out,
                    PreemptionMode.SWAP)
                if preempt_mode == PreemptionMode.SWAP:
                    running_result.swapped_out.append(sg)
                elif preempt_mode == PreemptionMode.RECOMPUTE:
                    running_result.preempted.append(sg)
            swaps_budget -= 1
            swapped_this_iter += 1
            force_preemption_count += 1
        return force_preemption_count, swapped_this_iter, swaps_budget

    def _build_process_list(self, running_result, swapped_this_iter):
        to_activate = []
        while len(to_activate) < swapped_this_iter and self._current_bin_queue:
            sg: SequenceGroup = self._current_bin_queue.popleft()
            if sg.first_seq.status == SequenceStatus.WAITING or \
                sg.first_seq.status == SequenceStatus.SWAPPED:
                to_activate.append(sg)

        running_in_bin = [
            sg for sg in list(self.running)
            if sg not in running_result.swapped_out
            and sg not in running_result.preempted
        ]
        min_tbt_required = min([sg.TBT_constraint for sg in running_in_bin],
                               default=0.001)

        process_list = to_activate + running_in_bin
        return process_list, min_tbt_required, to_activate, running_in_bin

    def _adjust_batch_limits(self, process_list, min_tbt_required, budget):
        if self.search_strategy != "sliding_window":
            return
        self.scheduler_config.max_num_seqs = max(self.max_batch_size,
                                                 len(process_list))
        budget.max_num_seqs = self.scheduler_config.max_num_seqs
        logger.info(
            f"Set max_num_seqs to {self.scheduler_config.max_num_seqs} for sliding window"
        )
        if (self.policy.interval_time is not None
                and self.policy.interval_time > min_tbt_required):
            self.scheduler_config.max_num_seqs -= self.max_swaps_per_iter
            logger.info(
                f"Adjust max_num_seqs to {self.scheduler_config.max_num_seqs} due to TBT {min_tbt_required} and policy interval {self.policy.interval_time}"
            )

    def _schedule_process_list(self, process_list, running_result,
                               prefill_result, swapped_in_result, budget,
                               enable_chunking):
        num_gpu_blocks_number = self.block_manager.get_max_gpu_num_blocks_number(
        )
        waiting_scheduled: List[Tuple[SequenceGroup, int, int]] = []
        swapped_scheduled: List[Tuple[SequenceGroup, int, int]] = []
        running_scheduled: List[Tuple[SequenceGroup, int, int]] = []
        stop_allocate = False
        stop_preempt = False
        force_preemption_count = 0

        for seq_group in process_list:
            seq_group_is_prefill = seq_group.is_prefill()

            if seq_group in self.waiting:
                seq_group_status = SequenceStatus.WAITING
            elif seq_group in self.swapped:
                seq_group_status = SequenceStatus.SWAPPED
            elif seq_group in self.running:
                seq_group_status = SequenceStatus.RUNNING
            else:
                continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(
                seq_group, seq_group_status, enable_chunking, budget)

            if seq_group_is_prefill:
                num_new_tokens = min(num_new_tokens, 2)

            not_update = False
            if num_new_tokens == 0:
                if len(seq_group.seqs) == 0:
                    logger.info(f"Sequence group status {seq_group_status}")
                    logger.info(
                        f"Sequence group {seq_group.request_id} has no sequence")
                    not_update = True
                elif seq_group.get_num_uncomputed_tokens() == 0:
                    not_update = True
                else:
                    stop_allocate = True

            if seq_group_status == SequenceStatus.WAITING:
                prompt_limit = self._get_prompt_limit(seq_group)
                if num_new_tokens > prompt_limit:
                    logger.warning(
                        "Input prompt (%d tokens) is too long and exceeds limit of %d",
                        num_new_tokens, prompt_limit)
                    for seq in seq_group.get_seqs(SequenceStatus.WAITING):
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    self.waiting.remove(seq_group)
                    prefill_result.ignored_seq_groups.append(seq_group)
                    continue

            num_lookahead_slots = self._get_num_lookahead_slots(
                seq_group_is_prefill, enable_chunking)
            num_blocks_number = self.block_manager.get_num_blocks_number(
                seq_group, seq_group_status, num_lookahead_slots)

            if (not stop_allocate and num_new_tokens > 0
                    and budget.can_schedule(num_new_tokens=num_new_tokens,
                                            num_new_seqs=num_new_seqs)
                    and num_blocks_number <= num_gpu_blocks_number):

                if seq_group_status == SequenceStatus.WAITING and stop_preempt:
                    continue

                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)
                num_gpu_blocks_number -= num_blocks_number

                if seq_group_status == SequenceStatus.RUNNING:
                    running_scheduled.append(
                        (seq_group, num_new_tokens, num_lookahead_slots))
                elif seq_group_status == SequenceStatus.WAITING:
                    if not stop_preempt:
                        waiting_scheduled.append(
                            (seq_group, num_new_tokens, num_lookahead_slots))
                else:
                    swapped_scheduled.append(
                        (seq_group, num_new_tokens, num_lookahead_slots))
                continue

            elif not not_update:
                stop_allocate = True

            if seq_group_status == SequenceStatus.RUNNING and num_new_tokens > 0:
                if seq_group.is_finished():
                    self._free_finished_seq_group(seq_group)
                    continue

                preempt_mode = self._preempt(
                    seq_group, running_result.blocks_to_swap_out,
                    PreemptionMode.SWAP)
                if preempt_mode == PreemptionMode.SWAP:
                    running_result.swapped_out.append(seq_group)
                elif preempt_mode == PreemptionMode.RECOMPUTE:
                    running_result.preempted.append(seq_group)
                force_preemption_count += 1

        for swapped_seq, num_new_tokens, num_lookahead_slots in swapped_scheduled:
            alloc_status = self.block_manager.can_swap_in(
                swapped_seq, num_lookahead_slots)
            if alloc_status != AllocStatus.OK:
                continue
            self.swapped.remove(swapped_seq)
            swapped_scheduled_result = self._schedule_slo_swapped(
                swapped_seq, num_lookahead_slots, enable_chunking)
            if swapped_scheduled_result.is_prefill:
                swapped_in_result.prefill_seq_groups.append(
                    swapped_scheduled_result.seq_group)
            else:
                swapped_in_result.decode_seq_groups.append(
                    swapped_scheduled_result.seq_group)
            swapped_in_result.blocks_to_swap_in.extend(
                swapped_scheduled_result.blocks_to_swap_in)
            swapped_in_result.blocks_to_copy.extend(
                swapped_scheduled_result.blocks_to_copy)

        for running_seq, num_new_tokens, num_lookahead_slots in running_scheduled:
            can_append = self._can_append_slots(running_seq, enable_chunking)
            if not can_append:
                running_result.leftover_running.append(running_seq)
                continue
            scheduled_seq_group: ScheduledSequenceGroup = self._scheduled_seq_group_cache[
                self.cache_id].get_object()
            scheduled_seq_group.seq_group = running_seq
            if running_seq.is_prefill():
                scheduled_seq_group.token_chunk_size = num_new_tokens
                running_result.prefill_seq_groups.append(scheduled_seq_group)
                running_result.prefill_seq_groups_list.append(running_seq)
            else:
                scheduled_seq_group.token_chunk_size = 1
                running_result.decode_seq_groups.append(scheduled_seq_group)
                running_result.decode_seq_groups_list.append(running_seq)
            self._append_slots(running_seq, running_result.blocks_to_copy,
                               enable_chunking)

        for waiting_seq, num_new_tokens, num_lookahead_slots in waiting_scheduled:
            remain_token_budget = budget.remaining_token_budget()
            if remain_token_budget > 0:
                more_new_tokens = 0
                for seq in waiting_seq.get_seqs(status=SequenceStatus.WAITING):
                    more_new_tokens += seq.get_num_new_tokens()
                more_new_tokens -= num_new_tokens
                add_tokens = min(more_new_tokens, remain_token_budget)
                num_new_tokens += add_tokens
                budget._num_batched_tokens += add_tokens

            can_allocate = self.block_manager.can_allocate(
                waiting_seq, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.NEVER:
                for seq in waiting_seq.get_seqs(status=SequenceStatus.WAITING):
                    seq.status = SequenceStatus.FINISHED_IGNORED
                prefill_result.ignored_seq_groups.append(waiting_seq)
                continue
            elif can_allocate == AllocStatus.LATER:
                continue

            self.waiting.remove(waiting_seq)
            waiting_scheduled_result = self._schedule_slo_prefills(
                waiting_seq, num_new_tokens, num_new_seqs, enable_chunking)
            prefill_result.seq_groups.append(waiting_scheduled_result.seq_group)

        return force_preemption_count

    def _schedule_preemption(
        self: "SLOScheduler",
        budget: "SchedulingBudget",
        enable_chunking: bool = False,
    ) -> "SLOSchedulerPreemptionOutputs":

        st_schedule_time = time.perf_counter()

        if self.search_strategy == "sliding_window":
            self._recompute_bins_sliding_window()
        elif not self._pending_swap_out_queue:
            self._recompute_bins()

        prefill_result, swapped_in_result, running_result = \
            self._init_preemption_outputs(enable_chunking, SchedulerPrefillOutputs,
                                          SchedulerSwappedInOutputs)

        if not self._exec_order and self.search_strategy in ["length_bin", None]:
            self.total_preemption_time += time.perf_counter() - st_schedule_time
            self._scheduler_running_outputs_cache[self.next_cache_id].reset()
            self._scheduled_seq_group_cache[self.next_cache_id].reset()
            return SLOSchedulerPreemptionOutputs(
                prefill_result=prefill_result,
                swapped_in_result=swapped_in_result,
                running_result=running_result,
                force_preemption_count=0,
            )

        self._inject_current_bin()

        preempted_count, swapped_this_iter, swaps_budget = self._preempt_pending(
            running_result)
        process_list, min_tbt_required, to_activate, running_in_bin = \
            self._build_process_list(running_result, swapped_this_iter)

        logger.info(
            f"process_list={len(process_list)} to_activate={len(to_activate)} running_in_bin={len(running_in_bin)} swapped_this_iter={swapped_this_iter} swaps_budget_left={swaps_budget} pending_swap_out_left={len(self._pending_swap_out_queue)}"
        )

        self._adjust_batch_limits(process_list, min_tbt_required, budget)

        force_preemption_count = preempted_count + self._schedule_process_list(
            process_list, running_result, prefill_result, swapped_in_result,
            budget, enable_chunking)

        if force_preemption_count > 0:
            logger.info(f"Force preemption count: {force_preemption_count}")
            for seq_group in running_result.preempted:
                logger.info(
                    f"Preempted sequence group {seq_group.collection_id} with RECOMPUTE mode"
                )
            for seq_group in running_result.swapped_out:
                logger.info(
                    f"Swapped out sequence group {seq_group.collection_id} with SWAP mode"
                )

        self.total_preemption_time += time.perf_counter() - st_schedule_time
        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        return SLOSchedulerPreemptionOutputs(
            prefill_result=prefill_result,
            swapped_in_result=swapped_in_result,
            running_result=running_result,
            force_preemption_count=force_preemption_count,
        )
    
    def _schedule_slo(self) -> "SchedulerOutputs":
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=max(len(self.running),
                             self.scheduler_config.max_num_seqs),
        )

        assert self.policy is not None
        self.free_finished_seq_groups()

        preemption_result: SLOSchedulerPreemptionOutputs = self._schedule_preemption(
            budget, enable_chunking=True)
        prefills = preemption_result.prefill_result
        swapped_in = preemption_result.swapped_in_result
        running_scheduled = preemption_result.running_result

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        # assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)

        # Update new running requests.
        # By default, vLLM scheduler prioritizes prefills.
        # Once chunked prefill is enabled,
        # the policy is changed to prioritize decode requests.
        self.running.clear()
        self.running.extend([s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend([s.seq_group for s in swapped_in.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend([s for s in running_scheduled.leftover_running])

        abort_seqs = self._stop_dull_scheduling()

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)

        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.prefill_seq_groups +
                                  swapped_in.prefill_seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups + abort_seqs,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )

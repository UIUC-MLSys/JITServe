import time
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from typing import (Callable, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)
from vllm.request_info import RequestType, RequestPhaseWeight
from vllm.sequence import SequenceGroup, Sequence, SequenceStatus
from vllm.logger import init_logger

logger = init_logger(__name__)

class BasePolicy(ABC):
    '''
    Base class for all scheduling policies.
    This class provides common methods for sequence group management,
    and defines an abstract method `get_priority`. Subclasses must implement 
    this method to calculate priority for sequence groups.
    '''
    def __init__(
        self,
        schedule_interval: int,
        penalty_factor: int = 1,
        window_size: int = 10,   # for interval calculation
    ) -> None:
        # Dictionary to store sequence groups by collection_id
        self.seq_group_dict: Dict[int, List["SequenceGroup"]] = {}
        self.seq_group_slo_dict: Dict[str, float] = {}
        self.collection_MADD_ratio: Dict[int, float] = {}
        
        # Record the time between two scheduling rounds
        self.num_schedule_count = 0
        self.schedule_interval = schedule_interval
        self.penalty_factor = penalty_factor
        
        # interval related
        self.last_schedule_time = None
        self.intervals: Deque[float] = deque(maxlen=window_size)
        self.interval_time = None  # average interval time

    def update_schedule_count(self) -> bool:
        '''
        Update the number of scheduler calls.
        '''
        self.num_schedule_count += 1
        if self.num_schedule_count % self.schedule_interval == 0:
            self.update_interval_time(time.time())
            self.seq_group_slo_dict = {}
            return True
        else:
            return False
        
    def update_interval_time(self, cur_time: float) -> None:
        '''
        Update the interval time between two scheduler calls.
        '''
        if self.last_schedule_time is None:
            self.last_schedule_time = cur_time
            self.intervals.append(0.02)
        else:
            new_interval = (cur_time - self.last_schedule_time) / self.schedule_interval
            self.intervals.append(new_interval)
            self.last_schedule_time = cur_time
        
        # 求平均
        if self.intervals:
            self.interval_time = sum(self.intervals) / len(self.intervals)
        else:
            self.interval_time = 0.02
            
     
    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        '''
        Add a sequence group to the policy.
        If the collection_id of the sequence group does not exist, a new list is created.
        '''
        if seq_group.collection_id not in self.seq_group_dict:
            self.seq_group_dict[seq_group.collection_id] = []
        self.seq_group_dict[seq_group.collection_id].append(seq_group)
    
    def delete_seq_group(self, seq_group: SequenceGroup) -> None:
        '''
        Remove a sequence group from the policy.
        If the collection_id corresponds to an empty list, the key-value pair is deleted.
        '''
        if seq_group.collection_id in self.seq_group_dict:
            if seq_group in self.seq_group_dict[seq_group.collection_id]:
                self.seq_group_dict[seq_group.collection_id].remove(seq_group)
            if len(self.seq_group_dict[seq_group.collection_id]) == 0:
                del self.seq_group_dict[seq_group.collection_id]
    
    @abstractmethod 
    def get_priority(self, seq_group: SequenceGroup) -> float:
        '''
        Calculate the priority of the given sequence group.
        This method must be implemented by subclasses.
        '''
        raise NotImplementedError
    
    @classmethod
    def _get_policy_cls(cls, policy_name: str) -> 'BasePolicy':
        policy_map = {
            "sjf": SJFPolicy,
            "srtf": SRTFPolicy,
            "concord": ConcordPolicy,
            "fcfs": FCFSPolicy,
            "las": LASPolicy,
        }

        # Check if policy name exists in the mapping
        policy_cls = policy_map.get(policy_name.lower())

        if policy_cls is None:
            raise ValueError(f"Unrecognized policy name: {policy_name}. Please provide a valid policy.")

        return policy_cls


class FCFSPolicy(BasePolicy):
    '''
    First-Come, First-Served (FCFS) scheduling policy.
    This policy schedules sequence groups based on the order of arrival.
    '''
    def __init__(
        self,
        schedule_interval: int = 20,
        penalty_factor: int = 1
    ) -> None:
        super().__init__(schedule_interval, penalty_factor)
        logger.info("FCFS policy is used")
    
    def get_priority(self, seq_group: SequenceGroup) -> float:
        '''
        Calculate the priority based on the order of arrival.
        '''
        return (0, seq_group.arrival_time)


class SJFPolicy(BasePolicy):
    '''
    Shortest Job First (SJF) scheduling policy.
    This policy calculates the priority based on the expected output length of the sequence group
    minus the actual output length generated so far. The job with the shortest expected duration
    is given the highest priority.
    '''
    def __init__(
        self,
        schedule_interval: int = 20,
        penalty_factor: int = 1
    ) -> None:
        super().__init__(schedule_interval, penalty_factor)
        logger.info("SJF policy is used")

    def get_priority(self, seq_group: SequenceGroup) -> float:
        '''
        Calculate the priority based on the expected output length minus the actual output length.
        '''
        seq_group.predict_output_length = seq_group.request_info.output_len
        return (0, seq_group.predict_output_length)


class SRTFPolicy(BasePolicy):
    '''
    Shortest Remaining Time First (SRTF) scheduling policy.
    This policy prioritizes sequence groups based on their deadline. Sequence groups with
    earlier deadlines are given higher priority.
    '''
    def __init__(
        self,
        schedule_interval: int = 20,
        penalty_factor: int = 1    
    ) -> None:
        super().__init__(schedule_interval, penalty_factor)
        logger.info("SRTF policy is used")
    
    def get_priority(self, seq_group: SequenceGroup) -> float:
        '''
        Calculate the priority based on the deadline of the sequence group.
        '''
        cur_time = time.time()
        time_to_deadline = seq_group.arrival_time + seq_group.deadline - cur_time
        if time_to_deadline > 0:
            return (0, time_to_deadline)
        else:
            return (1, -time_to_deadline)
        # return (0, cur_time - seq_group.arrival_time - seq_group.deadline)
    

class LASPolicy(BasePolicy):
    '''
    Least-Attained Service (LAS) scheduling policy.
    This policy prioritizes sequence groups based on the amount of service they have received so far.
    Sequence groups that have received less service are given higher priority.
    '''
    def __init__(
        self,
        schedule_interval: int = 20,
        penalty_factor: int = 1
    ) -> None:
        super().__init__(schedule_interval, penalty_factor)
        logger.info("LAS policy is used")
    
    def get_priority(self, seq_group: SequenceGroup) -> float:
        '''
        Calculate the priority based on the amount of service received so far.
        '''
        decode_len = seq_group.seqs[0].get_decode_len()
        service = decode_len
        if seq_group.request_type == RequestType.COLLECTIVE:
            for peer_seq_group in self.seq_group_dict[seq_group.collection_id]:
                peer_service = peer_seq_group.first_seq.get_decode_len()
                service += peer_service

        # Split service into discrete intervals, like 400 tokens
        service = int(service / 400) * 400
        return (service, seq_group.arrival_time)


class ConcordPolicy(BasePolicy):
    '''
    Service Level Objective (SLO) scheduling policy.
    This policy uses a weighted decay function to adjust the priority based on the real and expected output lengths.
    The goal is to maximize SLO (Service Level Objective) satisfaction.
    '''
    def __init__(
        self,
        schedule_interval: int = 50,
        penalty_factor: int = 1
    ) -> None:
        super().__init__(schedule_interval, penalty_factor)
        self.swap_in_time = 800
        self.swap_out_time = 800
        self.max_num_preemption_time = 20
        logger.info("SLO policy is used")
        
    def allocation_control(self, seq_group: SequenceGroup) -> float:
        total_remain_time = 0
        unit_time = self.interval_time if self.interval_time else 0.02
        #unit_time = seq_group.TBT_constraint
        for peer_seq_group in self.seq_group_dict[seq_group.collection_id]:
            peer_predict_output_len = peer_seq_group.predict_output_length
            peer_decode_len = peer_seq_group.seqs[0].get_decode_len()
            peer_remain_time = unit_time * (peer_predict_output_len - peer_decode_len)
            total_remain_time += peer_remain_time
            
        return total_remain_time
    
    def reward_estimation(self, seq_group: SequenceGroup) -> Tuple[float, float]:
        cur_time = time.time()
        input_len = seq_group.seqs[0].get_prompt_len()
        decode_len = seq_group.seqs[0].get_decode_len()
        predict_output_len = seq_group.request_info.output_len
        remain_len = predict_output_len - decode_len

        # time to deadline = deadline - (cur_time - arrival_time) = deadline - serve_time
        # finish_time = serve_time + remain_time
        unit_time = self.interval_time if self.interval_time else 0.02
        remain_time = unit_time * remain_len
        serve_time = cur_time - seq_group.arrival_time
        if seq_group.request_type == RequestType.COLLECTIVE:
            remain_time = max(self.allocation_control(seq_group), 0.001)
        else:
            remain_time = max(remain_time, 0.001)

        def _safe_ratio(numer: float, denom: float) -> float:
            """Clamp ratio between 1e-6 and 1.0 to avoid blowups."""
            return max(1e-6, min(1.0, numer / denom))

        reward = 0.0
        if seq_group.request_type == RequestType.LATENCY:
            # 如果已经有 TTFT 和 service_gain，说明进入 decode 阶段
            if seq_group.concord_metrics.TTFT is not None and seq_group.concord_metrics.service_gain > 0:
                predict_finish = serve_time + remain_time
                ratio = _safe_ratio(seq_group.deadline, predict_finish)
                decode_gain = remain_len * 8
                reward = seq_group.concord_metrics.service_gain + decode_gain * ratio**self.penalty_factor
            else:
                # prefill 阶段
                predict_finish_prefill = serve_time + unit_time
                predict_finish_decode = serve_time + remain_time
                predict_finish = predict_finish_prefill

                prefill_ratio = _safe_ratio(seq_group.TTFT_constraint, predict_finish_prefill)
                decode_ratio = _safe_ratio(seq_group.deadline, predict_finish_decode)

                prefill_gain = input_len * prefill_ratio**self.penalty_factor
                decode_gain = predict_output_len * decode_ratio**self.penalty_factor * 8
                reward = prefill_gain + decode_gain

        elif seq_group.request_type in (RequestType.THROUGHPUT, RequestType.COLLECTIVE):
            predict_finish = serve_time + remain_time
            ratio = _safe_ratio(seq_group.deadline, predict_finish)
            total_gain = input_len + predict_output_len * 8
            reward = total_gain * ratio**self.penalty_factor

        return reward, remain_time

    def get_priority(self, seq_group: SequenceGroup) -> float:
        """
        Calculate scheduling priority using Concord policy.
        Now uses dynamic `interval_time` instead of fixed TBT constraint.
        """

        # 如果之前算过就直接返回
        if self.seq_group_slo_dict.get(seq_group.request_id) is not None:
            return self.seq_group_slo_dict[seq_group.request_id]
        
        if seq_group.request_type == RequestType.LATENCY or seq_group.request_type == RequestType.THROUGHPUT:
            reward, remain_time = self.reward_estimation(seq_group)
            if abs(remain_time) < 1e-6:
                density = reward / 1e-6
            else:
                density = reward / remain_time
            concord_priority = (-density, remain_time)
            self.seq_group_slo_dict[seq_group.request_id] = concord_priority
            seq_group.slo_priority = concord_priority
        elif seq_group.request_type == RequestType.COLLECTIVE:
            # 计算 collective 的 priority
            total_reward = 0.0
            total_remain_time = 0.0
            for peer_seq_group in self.seq_group_dict[seq_group.collection_id]:
                peer_reward, peer_remain_time = self.reward_estimation(peer_seq_group)
                total_reward += peer_reward
                total_remain_time = peer_remain_time
            
            if abs(total_remain_time) < 1e-6:
                density = total_reward / 1e-6
            else:
                density = total_reward / total_remain_time
            concord_priority = (-density, total_remain_time)

            for peer_seq_group in self.seq_group_dict[seq_group.collection_id]:
                self.seq_group_slo_dict[peer_seq_group.request_id] = concord_priority
                peer_seq_group.slo_priority = concord_priority

        return concord_priority
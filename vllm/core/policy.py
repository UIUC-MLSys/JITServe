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
        interval_update_ratio: float = 0.5,
    ) -> None:
        # Dictionary to store sequence groups by collection_id
        self.seq_group_dict: Dict[int, List[SequenceGroup]] = {}    # collection_id             -> sequence_group
        self.seq_group_slo_dict: Dict[str, float] = {}              # sequence_group.request_id -> slo_gain
        self.collection_MADD_ratio: Dict[int, float] = {}           # collection_id             -> MADD_ratio
        
        # Record the time between two scheduling rounds
        self.num_schedule_count = 0
        self.schedule_interval = schedule_interval
        self.interval_update_ratio = interval_update_ratio
        
        self.last_schedule_time = time.time()
        self.interval_time = -1
        
    def update_schedule_count(self) -> bool:
        '''
        Update the number of scheduler calls.
        '''
        self.num_schedule_count += 1
        self.seq_group_slo_dict = {}
        if self.num_schedule_count % self.schedule_interval == 0:
            self.update_interval_time(time.time())
            return True
        else:
            return False
        
    def update_interval_time(self, cur_time: float) -> None:
        '''
        Update the interval time between two scheduler.
        '''
        if self.interval_time == -1:
            self.interval_time = 2
        else:
            self.interval_time = self.interval_time * self.interval_update_ratio + (1 - self.interval_update_ratio) * (cur_time - self.last_schedule_time)
        self.last_schedule_time = cur_time
     
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
    ) -> None:
        super().__init__(schedule_interval)
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
    ) -> None:
        super().__init__(schedule_interval)
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
    ) -> None:
        super().__init__(schedule_interval)
        logger.info("SRTF policy is used")
    
    def get_priority(self, seq_group: SequenceGroup) -> float:
        '''
        Calculate the priority based on the deadline of the sequence group.
        '''
        return (0, seq_group.deadline)


class ConcordPolicy(BasePolicy):
    '''
    Service Level Objective (SLO) scheduling policy.
    This policy uses a weighted decay function to adjust the priority based on the real and expected output lengths.
    The goal is to maximize SLO (Service Level Objective) satisfaction.
    '''
    def __init__(
        self,
        schedule_interval: int = 20,
    ) -> None:
        super().__init__(schedule_interval)
        self.swap_in_time = 800
        self.swap_out_time = 800
        self.max_num_preemption_time = 20
        logger.info("SLO policy is used")
        
    def allocation_control(self, seq_group: SequenceGroup) -> float:
        max_remain_time = 0
        for peer_seq_group in self.seq_group_dict[seq_group.collection_id]:
            peer_predict_output_len = peer_seq_group.predict_output_length
            peer_decode_len = peer_seq_group.seqs[0].get_decode_len()
            peer_remain_time = peer_seq_group.TBT_constraint * (peer_predict_output_len - peer_decode_len)
            max_remain_time = max(max_remain_time, peer_remain_time)
            
        return max_remain_time
    
    def get_priority(
        self, 
        seq_group: SequenceGroup
    ) -> float:
        '''
        Calculate the priority for the given sequence group based on the SLO policy.
        The priority is determined by the weighted decay of the real and expected output lengths,
        adjusted for the scheduling interval and task type.
        '''
        # avoid duplicate calculation
        if self.seq_group_slo_dict.get(seq_group.request_id) is not None:
            return self.seq_group_slo_dict[seq_group.request_id]
        
        # Notw(wei): update the predict output length if it is not default value (1024)
        seq_group.predict_output_length = seq_group.request_info.output_len
        
        cur_time = time.time()
        decode_len = seq_group.seqs[0].get_decode_len()
        input_len = seq_group.seqs[0].get_prompt_len()
        predict_output_len = seq_group.predict_output_length
        
        remain_time = seq_group.TBT_constraint * (predict_output_len - decode_len)
        if seq_group.request_type == RequestType.COLLECTIVE:
            remain_time = self.allocation_control(seq_group)
        remain_time = max(remain_time, 0.001)

        if seq_group.request_type == RequestType.LATENCY:
            if seq_group.concord_metrics.TTFT is not None and seq_group.concord_metrics.service_gain > 0:
                predict_finish_time = cur_time + seq_group.TBT_constraint * (predict_output_len - decode_len) - seq_group.arrival_time
                decode_gain = (predict_output_len - decode_len) * 2
                priority = seq_group.concord_metrics.service_gain + decode_gain * min(1, (seq_group.deadline / predict_finish_time)**2)
            else:
                predict_finish_prefill_time = cur_time + seq_group.TBT_constraint - seq_group.arrival_time
                predict_finish_decode_time = cur_time + seq_group.TBT_constraint * (predict_output_len - decode_len) - seq_group.arrival_time
                prefill_gain = input_len * min(1, (seq_group.TTFT_constraint / predict_finish_prefill_time)**2)
                decode_gain = predict_output_len * min(1, (seq_group.deadline / predict_finish_decode_time)**2) * 2
                priority = prefill_gain + decode_gain
        elif seq_group.request_type == RequestType.THROUGHPUT or seq_group.request_type == RequestType.COLLECTIVE:
            # priority = (seq_group.deadline - seq_group.TBT_constraint * (predict_output_len - decode_len)) + seq_group.arrival_time - cur_time
            predict_finish_time = cur_time + seq_group.TBT_constraint * (predict_output_len - decode_len) - seq_group.arrival_time
            total_gain = input_len * 1 + predict_output_len * 2
            priority = total_gain * min(1, (seq_group.deadline / predict_finish_time)**2)
        # if seq_group.request_type == RequestType.LATENCY:
        #     priority = (seq_group.deadline - seq_group.TBT_constraint / 2 * (predict_output_len - decode_len)) + seq_group.arrival_time - cur_time
        # elif seq_group.request_type == RequestType.THROUGHPUT or seq_group.request_type == RequestType.COLLECTIVE:
        #     priority = (seq_group.deadline - seq_group.TBT_constraint / 2 * (predict_output_len - decode_len)) + seq_group.arrival_time - cur_time
        priority = -priority / remain_time
        
        concord_priority = (priority, predict_output_len - decode_len)
        self.seq_group_slo_dict[seq_group.request_id] = concord_priority
        seq_group.slo_priority = concord_priority
        
        return concord_priority
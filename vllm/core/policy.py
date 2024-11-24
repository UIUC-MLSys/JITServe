import time
from abc import ABC, abstractmethod
from collections import deque
from typing import (Callable, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)
from vllm.request_info import RequestType
from vllm.sequence import SequenceGroup, Sequence
from vllm.logger import init_logger

logger = init_logger(__name__)

# List of supported scheduling policies
concord_support_policy = ["sjf", "srtf", "slo"]

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
        self.seq_group_dict = {}
        # Record the time between two scheduler
        self.num_schedule_count = 0
        self.schedule_interval = schedule_interval
        self.interval_update_ratio = interval_update_ratio
        self.last_schedule_time = time.perf_counter()
        self.interval_time = -1
        
    def update_schedule_count(self) -> bool:
        '''
        Update the number of scheduler calls.
        '''
        self.num_schedule_count += 1
        if self.num_schedule_count % self.schedule_interval == 0:
            self.update_interval_time(time.perf_counter())
            return True
        else:
            return False
        
    
    def update_interval_time(self, cur_time: float) -> None:
        '''
        Update the interval time between two scheduler.
        '''
        if self.interval_time == -1:
            self.interval_time = cur_time - self.last_schedule_time
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
        else:
            raise ValueError("Invalid sequence group")
    
    @abstractmethod 
    def get_priority(self, seq_group: SequenceGroup) -> float:
        '''
        Calculate the priority of the given sequence group.
        This method must be implemented by subclasses.
        '''
        raise NotImplementedError


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
        return seq_group.predict_output_length - seq_group.seqs[0].get_output_len()


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
        return seq_group.deadline


class SLOPoilicy(BasePolicy):
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
        logger.info("SLO policy is used")
        
    def soft_admission_control(self, 
                               seq_group: SequenceGroup,
                               cur_time: float) -> float:
        '''
        Perform soft admission control to determine whether the sequence group can be admitted.
        In this simplified version, it always returns True.
        '''
        return min(1, cur_time / seq_group.deadline)
    
    def weighted_decay(
        self, 
        req_type: RequestType, 
        real_length: int, 
        expe_length: int
    ) -> float:
        '''
        Calculate the weighted decay factor for a given request type.
        The decay factor is used to adjust the priority based on the ratio of real length to expected length.
        '''
        if req_type == RequestType.LATENCY:
            return (real_length / expe_length) ** 2  # Squared decay for latency
        elif req_type == RequestType.THROUGHPUT:
            return 1.0  # No decay for throughput
        # TODO
        elif req_type == RequestType.COLLECTIVE:
            return real_length / expe_length  # Linear decay for collective requests
        else:
            raise ValueError("Invalid request type")
    
    def get_priority(
        self, 
        seq_group: SequenceGroup
    ) -> float:
        '''
        Calculate the priority for the given sequence group based on the SLO policy.
        The priority is determined by the weighted decay of the real and expected output lengths,
        adjusted for the scheduling interval and task type.
        '''
        current_time = time.time()
        real_output_length = seq_group.seqs[0].get_output_len()
        expe_output_length = seq_group.get_expected_num_tokens(current_time)

        # For the next iteration, update the real and expected output lengths
        real_output_length_ne_iter = real_output_length + self.schedule_interval
        expe_output_length_ne_iter = seq_group.get_expected_num_tokens(current_time + self.interval_time)

        # Calculate SLO gain based on real and expected output lengths
        slo_gain = self.weighted_decay(seq_group.request_type, real_output_length, 
                                       expe_output_length) * real_output_length
        slo_gain_ne_iter = self.weighted_decay(seq_group.request_type, real_output_length_ne_iter,
                                                expe_output_length_ne_iter) * real_output_length_ne_iter
        
        # The change in SLO gain from the current iteration to the next iteration
        slo_gain = (slo_gain_ne_iter - slo_gain) * seq_group.request_weight.value
        slo_gain = self.soft_admission_control(seq_group, current_time) * slo_gain
        
        return -slo_gain
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import (Callable, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)
from vllm.request_info import RequestType, RequestTypeWeight, RequestPhaseWeight, service_compute
from vllm.sequence import SequenceGroup, Sequence
from vllm.logger import init_logger

logger = init_logger(__name__)

# List of supported scheduling policies
concord_support_policy = ["sjf", "srtf", "slo"]
# Prevent adjust ratio to be zero and cause division error
TINY_LIFT = 0.1

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
        self.seq_group_dict: Dict[int, List[SequenceGroup]] = {}        # collection_id -> sequence_group
        self.seq_group_slo_dict: Dict[str, float] = {}  # sequence_group.request_id -> slo_gain
        self.collection_MADD_ratio: Dict[int, float] = {}                # collection_id -> MADD_ratio
        
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
        if self.num_schedule_count % self.schedule_interval == 0:
            self.update_interval_time(time.time())
            self.seq_group_slo_dict = {}
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
        return seq_group.arrival_time


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
    
    def adjust_slo_priority(
        self, 
        seq_group: SequenceGroup,
        real_slo: float, 
        desire_slo: float
    ) -> float:
        '''
        Calculate the weighted decay factor for a given request type.
        The decay factor is used to adjust the priority based on the ratio of real length to expected length.
        '''
        agjust_ratio = None
        if seq_group.request_type == RequestType.LATENCY:
            # Here we consider the importance of deliver speed for latency-sensitive requests
            adjust_ratio = min(1, (real_slo / desire_slo))
        elif seq_group.request_type == RequestType.THROUGHPUT:
            # No decay for throughput-intensive requests
            adjust_ratio = 1.0  
        elif seq_group.request_type == RequestType.COLLECTIVE:
            # MADD (Minimum Allocation for Desired Duration) for collective requests
            if seq_group.collection_id in self.collection_MADD_ratio:
                return self.collection_MADD_ratio[seq_group.collection_id]
            
            real_serving_ratio = real_slo / seq_group.get_max_slo_gain()
            
            # Find the minimum serving ratio among all sequence groups in the same collection
            collection_serving_ratio = real_serving_ratio
            for collection_seq_group in self.seq_group_dict[seq_group.collection_id]:
                if collection_seq_group.request_id == seq_group.request_id:
                    continue
                collection_serving_ratio = min(collection_serving_ratio, 
                                               collection_seq_group.get_real_slo_gain() \
                                                / collection_seq_group.get_max_slo_gain())
            # The MADD ratio is the maximum of the desire serving ratio and the minimum collection serving ratio
            MADD_serving_ratio = max(desire_slo / seq_group.get_max_slo_gain(), 
                                       collection_serving_ratio)
            
            adjust_ratio = real_serving_ratio / MADD_serving_ratio    
        else:
            raise ValueError("Invalid request type")
        
        if adjust_ratio < TINY_LIFT:
            adjust_ratio = TINY_LIFT
            
        if seq_group.request_type == RequestType.COLLECTIVE:
            self.collection_MADD_ratio[seq_group.collection_id] = adjust_ratio
            
        return adjust_ratio
    
    def soft_admission_control(
        self, 
        seq_group: SequenceGroup,
        cur_time: float,
    ) -> float:
        '''
        Perform soft admission control to determine whether the sequence group can be admitted.
        In this simplified version, it always returns True.
        '''
        return min(1, cur_time / seq_group.deadline)
    
    def get_priority(
        self, 
        seq_group: SequenceGroup
    ) -> float:
        '''
        Calculate the priority for the given sequence group based on the SLO policy.
        The priority is determined by the weighted decay of the real and expected output lengths,
        adjusted for the scheduling interval and task type.
        '''
        # Fairness: 5% of the time, we schedule the sequence group based on the arrival time
        # This is to prevent starvation
        if self.num_schedule_count % (20 * self.schedule_interval) == 0:
            return seq_group.arrival_time
        if seq_group.request_id in self.seq_group_slo_dict:
            return self.seq_group_slo_dict[seq_group.request_id]
        
        current_time = self.last_schedule_time
        curr_prefill_len, curr_decode_len = seq_group.seqs[0].get_serving_len()
        # For the next scheduling round, we get the real serving length prediction
        # In normal case, the prefilling should be finished in the next scheduling round
        next_prefill_len = seq_group.seqs[0].get_prompt_len()
        next_decode_len = curr_decode_len + self.schedule_interval
        
        # If prefilling is not finished, then there will be another iteration used to finish the prefilling
        # So we need to adjust the real output length for the next scheduling round
        if curr_prefill_len < next_prefill_len:
            next_decode_len -= 1
        
        desire_prefill_len, desire_decode_len = seq_group.get_expected_num_tokens(current_time)

        # Calculate SLO based on real and expected lengths
        curr_slo = service_compute(curr_prefill_len, curr_decode_len)
        next_slo = service_compute(next_prefill_len, next_decode_len)
        desire_slo = service_compute(desire_prefill_len, desire_decode_len)
        
        # logger.info(f"Request Type: {seq_group.request_type}")
        # logger.info(f"Request {seq_group.request_id} SLO: {curr_slo} -> {next_slo} (Desire: {desire_slo})")
            
        # Weighted decay, this is for different request type
        delta_slo = (next_slo - curr_slo) * seq_group.request_weight.value
        
        # The change in SLO gain from the current iteration to the next iteration
        delta_slo =  delta_slo / self.adjust_slo_priority(seq_group, curr_slo, desire_slo)
        # logger.info(f"Adjusted SLO gain: {delta_slo}, {self.adjust_slo_priority(seq_group, curr_slo, desire_slo)}")
        delta_slo = delta_slo * self.soft_admission_control(seq_group, current_time) 
        
        # The priority is the negative of the change in SLO gain, for default sorted in ascending order
        self.seq_group_slo_dict[seq_group.request_id] = -delta_slo
        return -delta_slo
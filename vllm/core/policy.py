import time
from abc import ABC, abstractmethod
from collections import deque
from typing import (Callable, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)
from vllm.request_info import RequestType, RequestTypeWeight, RequestPhaseWeight, service_compute
from vllm.sequence import SequenceGroup, Sequence, SequenceStatus
from vllm.logger import init_logger

logger = init_logger(__name__)

# List of supported scheduling policies
concord_support_policy = ["sjf", "srtf", "slo"]
# Prevent adjust ratio to be zero and cause division error
TINY_LIFT = 0.1
PREEMPTION_OVERHEAD = 0.06
MAX_PRIORITY = 1e6

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
        # if self.num_schedule_count % 200 == 0:
        #     logger.info(f"Number of sequence groups: {len(self.seq_group_dict)}")
        #     seq_priortiy = [(request_id, priority) for request_id, priority in self.seq_group_slo_dict.items()]
        #     seq_priortiy = sorted(seq_priortiy, key=lambda x: x[1])
        #     
        #     if len(seq_priortiy) > 20:
        #         logger.info(f"Top 10 sequence groups: {seq_priortiy[:10]}")
        #         logger.info(f"Bottom 10 sequence groups: {seq_priortiy[-10:]}")
        #     else:
        #         logger.info(f"Sorted Sequence groups: {seq_priortiy}")
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
            self.interval_time = 0.06
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
        
    def measure_preemption_loss(self, running_service_gain) -> float:
        '''
        Calculate the priority for preemption.
        '''
        return time.time() + 1e6
    
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
        schedule_interval: int = 50,
    ) -> None:
        super().__init__(schedule_interval)
        self.swap_in_time = 800
        self.swap_out_time = 800
        self.max_num_preemption_time = 3
        logger.info("SLO policy is used")
    
    # def adjust_service_ratio(
    #     self, 
    #     seq_group: SequenceGroup,
    #     serve_time: float,
    #     deadline: float,
    # ) -> float:
    #     '''
    #     Calculate the weighted decay factor for a given request type.
    #     The decay factor is used to adjust the priority based on the ratio of real length to expected length.
    #     '''
    #     adjust_ratio = None
    #     if seq_group.request_type == RequestType.LATENCY:
    #         # Here we consider the importance of deliver speed for latency-sensitive requests
    #         adjust_ratio = min(1, (real_len / desire_len))
    #     elif seq_group.request_type == RequestType.THROUGHPUT:
    #         # No decay for throughput-intensive requests
    #         adjust_ratio = 1.0  
    #     elif seq_group.request_type == RequestType.COLLECTIVE:
    #         # MADD (Minimum Allocation for Desired Duration) for collective requests
    #         # if seq_group.collection_id in self.collection_MADD_ratio:
    #         #     return self.collection_MADD_ratio[seq_group.collection_id]
    #         
    #         real_alloc_ratio = real_len / seq_group.predict_output_length
    #         
    #         # Find the minimum serving ratio among all sequence groups in the same collection
    #         col_alloc_ratio = real_alloc_ratio
    #         for col_seq_group in self.seq_group_dict[seq_group.collection_id]:
    #             if col_seq_group.request_id == seq_group.request_id:
    #                 continue
    #             col_alloc_ratio = min(col_alloc_ratio, col_seq_group.seqs[0].get_output_len() \
    #                                             / col_seq_group.predict_output_length)
    #         # The MADD ratio is the maximum of the desire serving ratio and the minimum collection serving ratio
    #         MADD_serving_ratio = max(real_alloc_ratio / seq_group.get_max_slo_gain(), 
    #                                    collection_serving_ratio)
    #         
    #         adjust_ratio = real_serving_ratio / MADD_serving_ratio    
    #     else:
    #         raise ValueError("Invalid request type")
    #     
    #     if adjust_ratio < TINY_LIFT:
    #         adjust_ratio = TINY_LIFT
    #         
    #     # if seq_group.request_type == RequestType.COLLECTIVE:
    #     #     self.collection_MADD_ratio[seq_group.collection_id] = adjust_ratio
    #         
    #     return adjust_ratio
    
    def soft_admission_control(
        self, 
        seq_group: SequenceGroup,
        serve_time: float,
    ) -> float:
        '''
        Perform soft admission control to determine whether the sequence group can be admitted.
        In this simplified version, it always returns True.
        '''
        return min(1, (seq_group.deadline / serve_time)**2)
    
    def measure_preemption_loss(self, running_service_gain) -> float:
        '''
        Calculate the priority for preemption.
        '''
        logger.info(f"running_service_gain {running_service_gain}")
        logger.info(f"interval_time {self.interval_time}")
        return running_service_gain / self.interval_time * PREEMPTION_OVERHEAD
    
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
        
        # update the predict output length if it is not default value (1024)
        if seq_group.request_info.output_len != 1024:
            seq_group.predict_output_length = seq_group.request_info.output_len
        
        if seq_group.num_cumulative_preemption > self.max_num_preemption_time:
            self.seq_group_slo_dict[seq_group.request_id] = (-MAX_PRIORITY, seq_group.arrival_time)
            return (-MAX_PRIORITY, seq_group.arrival_time)
        
        cur_time = self.last_schedule_time
        
        real_input_len, real_output_len = seq_group.seqs[0].get_serving_len()
        expected_input_len, expected_output_len = seq_group.get_expected_num_tokens(cur_time)
        
        # Compute the delta_input_token and delta_output_token
        max_input_len = seq_group.seqs[0].get_prompt_len()
        max_output_len = seq_group.predict_output_length if seq_group.predict_output_length > real_output_len \
            else real_output_len + self.schedule_interval
        delta_input_token = max_input_len - real_input_len
        delta_output_token = min(max_output_len - real_output_len, self.schedule_interval)
        
        is_latency = seq_group.request_type == RequestType.LATENCY
        delta_service = service_compute(delta_input_token, delta_output_token, is_latency,
                                        real_input_len, real_output_len, expected_input_len, expected_output_len)
        
        serve_time = (self.last_schedule_time - seq_group.arrival_time) * 1000
        sigma = self.soft_admission_control(seq_group, serve_time)
        
        concord_priority = (-delta_service * sigma, seq_group.arrival_time)
        # concord_priority = (0, seq_group.arrival_time)
        self.seq_group_slo_dict[seq_group.request_id] = concord_priority
        seq_group.slo_priority = concord_priority
        return concord_priority
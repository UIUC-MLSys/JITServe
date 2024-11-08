import asyncio
import uuid
from collections import deque
from typing import List, Optional, Dict
import torch
import time
import numpy as np
from typing import Deque
import math

from vllm.logger import init_logger
from vllm.sequence import SequenceGroup
from vllm.config import _EMBEDDING_MODEL_MAX_NUM_BATCHED_TOKENS

logger = init_logger(__name__)

def calculate_time(show=False, min_cost_ms=0.0):
    def wrapper(func):
        def inner_func(*args, **kwargs):
            torch.cuda.synchronize()
            if show:
                start_time = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            if show:
                cost_time = (time.time() - start_time) * 1000
                if cost_time > min_cost_ms:
                    print(f"Function {func.__name__} took {cost_time} ms to run.")
            return result

        return inner_func

    return wrapper


class VTCReqQueue:

    def __init__(self, 
                 num_gpu_blocks,
                 block_size,
                 max_num_batched_tokens, 
                 max_num_seqs,
                 max_model_len, 
                 input_price=1, output_price=2) -> None:
        self.num_gpu_blocks=num_gpu_blocks
        self.block_size=block_size
        assert max_num_batched_tokens is not None
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.waiting_req_list: List[SequenceGroup] = []
        self.input_price = input_price
        self.output_price = output_price
        # client_id: counter value
        self.served: Dict[int, int] = {}
        # client_id: list of sequences(requests)
        self.user_req_list: Dict[int, List[SequenceGroup]] = {}

    def append(self, req: SequenceGroup):
        self.waiting_req_list.append(req)
        if req.client_id not in self.user_req_list:
            self.user_req_list[req.client_id] = deque([req])
            self.served[req.client_id] = 0
        else:
            self.user_req_list[req.client_id].append(req)

        # waiting queue was empty before
        if len(self.user_req_list[req.client_id]) == 1:
            # lift counter
            cnts = [v for k, v in self.served.items()
                      if (len(self.user_req_list[k]) > 0 and k != req.client_id)]
            if len(cnts) > 0:
                self.served[req.client_id] = max(self.served[req.client_id], min(cnts))


    def _init_cache_list(self, current_batch:Deque[SequenceGroup]):
        self.cache_len_list = []
        if len(current_batch) > 0:
            for req in current_batch:
                # (current num of tokens, max remaining tokens)
                sequence_len = req.first_seq.get_prompt_len + req.first_seq.get_output_len
                self.cache_len_list.append((sequence_len, self.max_model_len - sequence_len))

    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req: SequenceGroup):
        self.cache_len_list.append(
            (req.first_seq.get_prompt_len + 1, self.max_model_len - req.first_seq.get_prompt_len - 1)
        )
        # Sort cache_len_list in descending order based on remaining length
        self.cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = [[e[1] for e in self.cache_len_list]]
        has_run_len_array = [[e[0] for e in self.cache_len_list]]

        max_required_blocks = 0
        num_of_seq = len(self.cache_len_list)

        for i in range(num_of_seq-1, -1, -1):
            num_iter = left_out_len_array[i] # num of iterations passed by
            left_out_len_array -= num_iter
            has_run_len_array += num_iter
            current_block = sum([math.ceil(e/self.block_size) for e in has_run_len_array[:i+1]])
            max_required_blocks = max(current_block, max_required_blocks)

        # Compare the maximum block requirement with available GPU blocks
        if max_required_blocks <= self.num_gpu_blocks:
            return True
        else:
            return False


    def generate_new_batch(self,
                           current_batch:Deque[SequenceGroup]
                           ) -> Optional[Deque[SequenceGroup]]:
        if current_batch is not None and len(current_batch) >= self.max_num_seqs:
            return None
        if len(self.served) == 0:
            return None
        
        self._init_cache_list(current_batch)
        can_run_list = deque()
        # abort_list = []
        new_batch_total_tokens = 0
        # aborted_count = 0
        active_served = {k: v for k, v in self.served.items()}
        while True:
            if len(active_served) == 0:
                break
            client_id = min(active_served, key=active_served.get)
            if len(self.user_req_list[client_id]) > 0:
                req = self.user_req_list[client_id][0]
                # if req.aborted:
                #     aborted_count += 1
                #     abort_list.append(req)
                #     self.user_req_list[client_id].popleft()
                #     continue
                if (self._can_add_new_req(req) and
                    new_batch_total_tokens + req.first_seq.get_prompt_len <= self.max_num_batched_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.first_seq.get_prompt_len
                    self.user_req_list[client_id].popleft()
                    # update fairness counter, adopt linear cost function in VTC
                    self.served[client_id] += req.first_seq.get_prompt_len * self.input_price
                    active_served[client_id] += req.first_seq.get_prompt_len * self.input_price
                else:
                    break
            else:
                del active_served[client_id]

        if len(can_run_list) != 0:
            # self.waiting_req_list = [req for req in self.waiting_req_list
            #                          if req not in can_run_list and req not in abort_list]
            self.waiting_req_list = [req for req in self.waiting_req_list
                                     if req not in can_run_list]
            return can_run_list
        else:
            return None

    
    def update_counter(self, current_batch: Deque[SequenceGroup]):
        for req in current_batch:
            self.served[req.client_id] += 1 * self.output_price

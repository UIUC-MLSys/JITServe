import json
import numpy as np
import pandas as pd
import random
import os
import sys
from dataclasses import dataclass 
from enum import Enum, IntEnum
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Tuple

from jitserve.request_info import RequestInfo, RequestType

support_datasets = ['alpaca', 'lmsys_chat', 'ToT', 'mix']
throughput_hint_words = ['code', 'function', 'method', 'class', 'variable', 'python', 'test']
collective_prompt = "Given the following question, try to reason it step by step and give several possible answers. The question is:"
default_burst_gpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "traces", "burst", "BurstGPT_1.csv")

class BaseDataset:
    '''
    A base class to represent a trace of requests.
    Note: Datasets are different for different models.
    '''
    def __init__(
        self, 
        prompts: List[str],
        outputs: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_request_num: int = 1000,
        is_real: bool = False,
        is_random_pick: bool = False,
        poisson_lambda: float = 0.9,
        deadline_range: Tuple[int, int] = (1000, 50000),
        ) -> None:
           
        assert len(prompts) == len(outputs)
        self.prompts = prompts
        self.outputs = outputs
        
        self.max_request_num = max_request_num
        if len(prompts) > self.max_request_num:
            self.prompts = prompts[:self.max_request_num]
            self.outputs = outputs[:self.max_request_num]
        
        self.tokenizer = tokenizer
        
        self.is_random_pick = is_random_pick
        self.is_real = is_real
        self.poisson_lambda = poisson_lambda
        self.deadline_range = deadline_range
        
        self.prompts_len = self._get_prompt_length()
        self.outputs_len = self._get_output_length()
        self.request_type_list = self._categorize_dataset()
        self.deliver_time = self._generate_time_stamp()
        self.deadline = self._generate_deadline()
        self.priority = self._generate_priority()
        
        self.request_list = self._construct_request_list()
        
        
    def _get_prompt_length(self) -> List[int]:
        '''
        Get the token length of each prompt.
        '''
        print("Calculating the token length of each prompt...")
        prompts_length = []
        for prompt in tqdm(self.prompts):
            prompts_length.append(len(self.tokenizer(prompt).input_ids))
        return prompts_length

    def _get_output_length(self) -> List[int]:
        '''
        Get the token length of each output.
        '''
        print("Calculating the token length of each output...")
        outputs_length = []
        for output in tqdm(self.outputs):
            outputs_length.append(len(self.tokenizer(output).input_ids))
        return outputs_length
        

    def _categorize_dataset(self) -> List[RequestType]:
        '''
        Categorize the dataset into three categories: throughput, latency, and collective.
        For collective requests, we will add a prefix to the prompt.
        '''
        num_throughput = 0
        num_latency = 0
        num_collective = 0
        request_type_list = []
        
        if self.is_random_pick:
            throughput_request_ratio = 0.2
            latency_request_ratio = 0.6
            num_requests = len(self.prompts)
            
            num_throughput = int(num_requests * throughput_request_ratio)
            num_latency = int(num_requests * latency_request_ratio)
            num_collective = num_requests - num_throughput - num_latency

            request_type_list = (
                [RequestType.THROUGHPUT] * num_throughput +
                [RequestType.LATENCY] * num_latency +
                [RequestType.COLLECTIVE] * num_collective
            )
            random.shuffle(request_type_list)
        else:
            for prompt in self.prompts:
                if any(word in prompt for word in throughput_hint_words):
                    prompt_type = RequestType.THROUGHPUT
                elif random.random() < 0.25:
                    prompt_type = RequestType.COLLECTIVE
                else:
                    prompt_type = RequestType.LATENCY
                request_type_list.append(prompt_type)

                if prompt_type == RequestType.THROUGHPUT:
                    num_throughput += 1
                elif prompt_type == RequestType.LATENCY:
                    num_latency += 1
                else:
                    num_collective += 1
                    
        for i, prompt in enumerate(self.prompts):
            if request_type_list[i] == RequestType.COLLECTIVE:
                self.prompts[i] = collective_prompt + prompt
        
        # Print the number of requests in each category
        print(f"Number of throughput requests: {num_throughput}, ratio: {num_throughput / len(self.prompts)}")
        print(f"Number of latency requests: {num_latency}, ratio: {num_latency / len(self.prompts)}")
        print(f"Number of collective requests: {num_collective}, ratio: {num_collective / len(self.prompts)}")
        print(f"Total number of requests: {len(self.prompts)}")
        
        return request_type_list
        
        
    def _generate_time_stamp(self) -> List[int]:
        '''
        Generate the timestamp for each request.
        '''
        deilver_time = []
        
        if self.is_real:
            burstGPT_data = pd.read_csv(default_burst_gpt_path)
            timestamp = burstGPT_data['Timestamp'].tolist()
            deliver_time = timestamp[:len(self.prompts)]    
        else:
            # Generate the deliver time and deadline for each request
            # The deliver time is generated under poisson distribution
            deliver_time = np.cumsum(np.random.poisson(self.poisson_lambda, len(self.prompts)))
            
        return deliver_time
        
        
    def _generate_deadline(self) -> List[int]:
        '''
        Generate the deadline for each request.
        '''
        assert self.deadline_range[0] < self.deadline_range[1]
        deadline = [self.deliver_time[i] + random.randint(self.deadline_range[0], self.deadline_range[1]) \
                                for i in range(len(self.deliver_time))]
        return deadline
    
    
    def _generate_priority(self) -> List[int]:
        '''
        Generate the priority for each request.
        '''
        priority = [0 if random.random() < 0.9 else 1 for _ in range(len(self.prompts))]
        return priority
    
    
    def _construct_request_list(self) -> List[RequestInfo]:
        '''
        Construct the request list.
        '''
        request_list = []
        for i in range(len(self.prompts)):
            # Create RequestInfo with proper parameters
            default_slo_constraint = (1000.0, 1000.0, 5000.0)  # ttft, tbt, ttlt
            request = RequestInfo(
                request_type=self.request_type_list[i],
                slo_constraint=default_slo_constraint,
                client_id=0,  # Default client ID
                collection_id=i,
                deadline=int(self.deadline[i]),
                input_len=self.prompts_len[i],
                output_len=self.outputs_len[i],
                prediction_task=None,
                prompt=self.prompts[i],
                output=self.outputs[i],
                stage_id=0,  # Default stage ID for non-collective requests
                request_id=i,
                state=""  # Default state
            )
            request_list.append(request)
        return request_list
    
    
    def to_json(self) -> Dict:
        '''
        Convert the dataset to a dictionary.
        '''
        dataset = [{
            "prompt": request.prompt,
            "output": request.output,
            "prompt_len": request.prompt_len,
            "output_len": request.output_len,
            "collection_id": request.collection_id,
            "request_type": request.request_type,
            "deliver_time": request.deliver_time,
            "deadline": request.deadline,
            "priority": request.priority,
        } for request in self.request_list]
        return dataset
    
    @classmethod
    def divide_by_rate(cls, dataset: List[RequestInfo], rate: List[float]) -> List[List[RequestInfo]]:
        '''
        Divide the dataset into several parts by the rate.
        '''
        assert sum(rate) == 1
        # shuffle the dataset
        random.shuffle(dataset)
        num_requests = len(dataset)
        num_requests_list = [int(num_requests * r) for r in rate]
        num_requests_list[-1] = num_requests - sum(num_requests_list[:-1])
        
        divided_dataset = []
        start_idx = 0
        for num in num_requests_list:
            divided_dataset.append(dataset[start_idx: start_idx + num])
            start_idx += num
            
        [dataset.sort(key=lambda x: x.deliver_time) for dataset in divided_dataset]
        
        return divided_dataset
    
   
class TraceConfig:
    '''
    A class to represent the configuration of a trace.
    '''
    def __init__(self, **kwargs):
        # Set default values
        self.model = kwargs.get('model', None)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.dataset_name = kwargs.get('dataset_name', 'lmsys_chat')
        self.dataset_path = kwargs.get('dataset_path', "./dataset/")
        self.max_request_num = kwargs.get('max_request_num', 1000)
        self.is_real = kwargs.get('is_real', False)
        self.is_random_pick = kwargs.get('is_random_pick', False)
        self.trust_model_code = kwargs.get('trust_model_code', False)
        self.poisson_lambda = kwargs.get('poisson_lambda', 80)
        self.deadline_range = kwargs.get('deadline_range', (1000, 50000))
        self.seed = kwargs.get('seed', 42)
    
    @classmethod
    def from_dict(cls, config_dict):
        '''
        Initialize TraceConfig from a dictionary.
        '''
        return cls(**config_dict)
    
    def __str__(self):
        return f"""
                Model: {self.model}, Tokenizer: {self.tokenizer}, Dataset Name: {self.dataset_name},
                Dataset Path: {self.dataset_path}, Max Request Number: {self.max_request_num}, Is Real: {self.is_real},
                Is Random Pick: {self.is_random_pick}, Trust Model Code: {self.trust_model_code}, Poisson Lambda: {self.poisson_lambda},
                Deadline Range: {self.deadline_range}, Seed: {self.seed}
                """
                
                
class Trace:
    '''
    A class to represent a trace of requests.
    '''
    def __init__(
        self,
        model: str,
        tokenizer: str = None,
        dataset_name: str = None,
        dataset_path: str = "./dataset/",
        max_request_num: int = 1000,
        is_real: bool = False,
        is_random_pick: bool = False,
        trust_model_code: bool = False,
        poisson_lambda: float = 80,
        deadline_range: Tuple[int, int] = (1000, 50000),
        seed: int = 42,
    ):
        assert model is not None
        self.model = model
        print(f"Using the model: {model}")
        print(f"Using the dataset: {dataset_name}")
        
        if tokenizer is not None:
            print(f"Using the tokenizer: {tokenizer}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_model_code=trust_model_code)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_model_code=trust_model_code)
        
        self.seed = seed
        self.max_request_num = max_request_num
        self.is_real = is_real
        self.is_random_pick = is_random_pick
        self.poisson_lambda = poisson_lambda
        self.deadline_range = deadline_range
        
        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Construct the dataset, we can either pass the dataset or the dataset_path
        if dataset_name is None and dataset_path is not None:
            dataset_name: str = dataset_path.split('/')[-1].replace('.json', '')
            
        assert dataset_name in support_datasets
        
        if dataset_name == 'alpaca':
            self.dataset = self.load_alpaca(dataset_path)
        elif dataset_name == 'lmsys_chat':
            self.dataset = self.load_lmsys_chat(dataset_path)
            
    
    @classmethod
    def from_config(cls, config: TraceConfig) -> 'Trace':
        '''
        Initialize Trace from a configuration object.
        '''
        return cls(**config.__dict__)
    
    
    def load_alpaca(
        self, 
        dataset_path: str = None, 
    ) -> BaseDataset:
        if dataset_path is None:
            dataset_path = f"./dataset/{self.model}/alpaca.json"
            
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            prompts = []
            outputs = []
            
            for data in dataset:
                assert data["prompt"] is not None
                assert data["output"] is not None
                
                # Sanity check, there should not be data with more than 2 keys
                if len(data) > 2:
                    print(data.keys())
                    
                prompt = data["prompt"]
                prompts.append(prompt)
                output = data["output"]
                outputs.append(output)

                
        return BaseDataset(prompts, outputs, self.tokenizer, self.max_request_num, self.is_real,
                           self.is_random_pick, self.poisson_lambda, self.deadline_range)
    
    
    def load_lmsys_chat(
        self, 
        dataset_path: str = None, 
    ) -> BaseDataset:
        if dataset_path is None:
            dataset_path = f"./dataset/{self.model}/lmsys_chat.json"
            
        with open(dataset_path) as f:
            dataset = json.load(f)
            prompts = []
            outputs = []
            
            for data in dataset:
                assert data["prompt"] is not None
                assert data["output"] is not None
                
                # Note: data with more than 2 keys
                if len(data) > 2:
                    print(data.keys())
                    
                prompt = data["prompt"]
                prompts.append(prompt)
                output = data["output"]
                outputs.append(output)

                
        return BaseDataset(prompts, outputs, self.tokenizer, self.max_request_num, self.is_real,
                           self.is_random_pick, self.poisson_lambda, self.deadline_range)
    
    
    def load_APPS(
        self, 
        dataset_path: str = None, 
    ) -> BaseDataset:
        if dataset_path is None:
            dataset_path = f"./dataset/{self.model}/APPS.json"
            
        with open(dataset_path) as f:
            dataset = json.load(f)
            prompts = []
            outputs = []
            
            for data in dataset:
                assert data["prompt"] is not None
                assert data["output"] is not None
                
                # Note: data with more than 2 keys
                if len(data) > 2:
                    print(data.keys())
                    
                prompt = data["prompt"]
                prompts.append(prompt)
                output = data["output"]
                outputs.append(output)

                
        return BaseDataset(prompts, outputs, self.tokenizer, self.max_request_num, self.is_real,
                           self.is_random_pick, self.poisson_lambda, self.deadline_range)
        
        
    def save_trace(self, save_path: str) -> None:
        '''
        Save the trace to a json file.
        '''
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset.to_json(), f, indent=4, ensure_ascii=False)
            
    
    @classmethod
    def load_trace(cls, trace_path: str) -> List['RequestInfo']:
        '''
        Load the trace from a json file.
        '''
        
        with open(trace_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        request_list = []
        for req in dataset:
            # Convert trace format to RequestInfo
            request_info = RequestInfo.from_json(req)
            request_list.append(request_info)
        return request_list

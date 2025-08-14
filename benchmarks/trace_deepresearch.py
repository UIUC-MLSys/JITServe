"""DeepResearch trace handling for collective requests."""
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class RequestType(Enum):
    LATENCY = 0
    THROUGHPUT = 1
    COLLECTIVE = 2


@dataclass
class DeepResearchRequest:
    """Represents a single request within a stage."""
    prompt: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    state: str
    timestamp: float
    output: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeepResearchRequest':
        return cls(
            prompt=data.get('prompt', ''),
            input_tokens=data.get('input_tokens', 0),
            output_tokens=data.get('output_tokens', 0),
            total_tokens=data.get('total_tokens', 0),
            state=data.get('state', ''),
            timestamp=data.get('timestamp', 0.0),
            output=data.get('output', None)
        )


@dataclass
class DeepResearchStage:
    """Represents a stage containing one or more requests."""
    requests: List[DeepResearchRequest]
    
    @classmethod
    def from_list(cls, stage_data: List[Dict[str, Any]]) -> 'DeepResearchStage':
        requests = [DeepResearchRequest.from_dict(req) for req in stage_data]
        return cls(requests=requests)


@dataclass
class DeepResearchCollectiveRequest:
    """Represents a complete collective request with multiple stages."""
    stage_num: int
    stages: List[DeepResearchStage]
    collection_id: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], collection_id: int) -> 'DeepResearchCollectiveRequest':
        stage_num = data.get('stage_num', 0)
        stages_data = data.get('stages', [])
        stages = [DeepResearchStage.from_list(stage) for stage in stages_data]
        return cls(
            stage_num=stage_num,
            stages=stages,
            collection_id=collection_id
        )
    
    def to_request_formats(self) -> List['RequestFormat']:
        """Convert to RequestFormat objects for each request in all stages."""
        request_formats = []
        for stage_idx, stage in enumerate(self.stages):
            for req_idx, request in enumerate(stage.requests):
                request_format = RequestFormat(
                    prompt=request.prompt,
                    output=request.output or '',
                    prompt_len=request.input_tokens,
                    output_len=request.output_tokens,
                    request_type=RequestType.COLLECTIVE,
                    collection_id=self.collection_id,
                    stage_id=stage_idx,
                    request_id=req_idx,
                    state=request.state,
                    timestamp=request.timestamp,
                    deadline=0  # Will be set based on SLO constraints
                )
                request_formats.append(request_format)
        return request_formats


class RequestFormat:
    """Standard request format compatible with existing system."""
    def __init__(
        self,
        prompt: str,
        output: str,
        prompt_len: int,
        output_len: int,
        request_type: RequestType,
        collection_id: int = 0,
        stage_id: int = 0,
        request_id: int = 0,
        state: str = '',
        timestamp: float = 0.0,
        deadline: float = 0.0
    ):
        self.prompt = prompt
        self.output = output
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.request_type = request_type
        self.collection_id = collection_id
        self.stage_id = stage_id
        self.request_id = request_id
        self.state = state
        self.timestamp = timestamp
        self.deadline = deadline
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prompt': self.prompt,
            'output': self.output,
            'prompt_len': self.prompt_len,
            'output_len': self.output_len,
            'request_type': self.request_type.value,
            'collection_id': self.collection_id,
            'stage_id': self.stage_id,
            'request_id': self.request_id,
            'state': self.state,
            'timestamp': self.timestamp,
            'deadline': self.deadline
        }


class DeepResearchTrace:
    """Handles loading and processing DeepResearch trace files."""
    
    @staticmethod
    def load_trace(trace_path: str) -> List[DeepResearchCollectiveRequest]:
        """Load DeepResearch trace from JSONL file."""
        trace_file = Path(trace_path)
        if not trace_file.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")
        
        collective_requests = []
        with open(trace_file, 'r') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    collective_request = DeepResearchCollectiveRequest.from_dict(data, collection_id=idx)
                    collective_requests.append(collective_request)
        
        return collective_requests
    
    @staticmethod
    def flatten_to_request_formats(
        collective_requests: List[DeepResearchCollectiveRequest]
    ) -> List[RequestFormat]:
        """Flatten all collective requests into a list of RequestFormat objects."""
        all_requests = []
        for collective_req in collective_requests:
            all_requests.extend(collective_req.to_request_formats())
        return all_requests
    
    @staticmethod
    def get_stage_structure(
        collective_request: DeepResearchCollectiveRequest
    ) -> Tuple[int, List[int]]:
        """Get the structure of stages (number of stages and requests per stage)."""
        num_stages = collective_request.stage_num
        requests_per_stage = [len(stage.requests) for stage in collective_request.stages]
        return num_stages, requests_per_stage
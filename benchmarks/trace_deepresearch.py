"""DeepResearch trace handling for collective requests."""
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Import RequestInfo from vllm.request_info
import sys
sys.path.append('../')
from vllm.request_info import RequestInfo


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
    
    def to_request_infos(self) -> List[RequestInfo]:
        """Convert to RequestInfo objects for each request in all stages."""
        request_infos = []
        for stage_idx, stage in enumerate(self.stages):
            for req_idx, request in enumerate(stage.requests):
                # Create default SLO constraint (will be overridden later)
                default_slo_constraint = (1000.0, 1000.0, 5000.0)  # ttft, tbt, ttlt
                
                request_info = RequestInfo(
                    request_type=RequestType.COLLECTIVE,
                    slo_constraint=default_slo_constraint,
                    client_id=0,  # Will be set later
                    collection_id=self.collection_id,
                    deadline=0,  # Will be set based on SLO constraints
                    input_len=request.input_tokens,
                    output_len=request.output_tokens,
                    prediction_task=None,
                    prompt=request.prompt,
                    output=request.output or '',
                    stage_id=stage_idx,
                    request_id=req_idx,
                    state=request.state
                )
                request_infos.append(request_info)
        return request_infos
    
    def get_stage_structure(self) -> Tuple[int, List[int]]:
        """Get the structure of stages (number of stages and requests per stage)."""
        num_stages = self.stage_num
        requests_per_stage = [len(stage.requests) for stage in self.stages]
        return num_stages, requests_per_stage



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
    def flatten_to_request_infos(
        collective_requests: List[DeepResearchCollectiveRequest]
    ) -> List[RequestInfo]:
        """Flatten all collective requests into a list of RequestInfo objects."""
        all_requests = []
        for collective_req in collective_requests:
            all_requests.extend(collective_req.to_request_infos())
        return all_requests
    

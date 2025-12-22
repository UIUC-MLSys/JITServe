import asyncio
from enum import Enum, IntEnum
from typing import List, Dict, Any, Tuple

# Enum class to define different types of requests
class RequestType(IntEnum):
    LATENCY = 0  # Latency-sensitive request
    THROUGHPUT = 1  # Throughput-sensitive request
    COLLECTIVE = 2  # Collective request (e.g., batch processing)

class RequestTypeWeight(Enum):
    LOW = 1  # Low priority
    HIGH = 2  # High priority
    
class RequestPhaseWeight(Enum):
    PREFILL = 1
    DECODE = 8

class RequestApplication(Enum):
    TOT = 1
    DEEPRESEARCH = 2
    
class RequestDeltaInfo:
    def __init__(
        self,
        cur_time: float,
        delta_time: float,
        delta_prefill_len: int,
        delta_decode_len: int
    ) -> None:
        self.cur_time = cur_time 
        self.delta_time = delta_time
        self.delta_prefill_len = delta_prefill_len
        self.delta_decode_len = delta_decode_len   
    
# Class to hold information about a request
class RequestInfo:
    '''
    Class that stores information about a request such as its type, client, collection,
    deadline, output length, and priority weight. 
    This class is used to encapsulate the request data for scheduling and prioritization.
    '''
    def __init__(
        self, 
        request_type: RequestType,  # Type of the request (LATENCY, THROUGHPUT, or COLLECTIVE)
        slo_constraint: Tuple[float, float, float],  # SLO constraint for the request (ttft, tbt, ttlt)
        client_id: int,  # ID of the client making the request
        collection_id: int,  # ID of the collection the request belongs to
        deadline: int,  # Deadline for the request to be processed
        input_len: int,  # The input length for the request
        output_len: int,  # The expected output length for the request
        prediction_task: asyncio.Task = None,  # Task for prediction
        prompt: str = "",  # The prompt text for the request
        output: str = "",  # The output text for the request
        stage_id: int = 0,  # Stage ID for collective requests
        request_id: int = 0,  # Request ID within a stage
        state: str = "",  # State information for the request
    ):
        # Initialize the attributes with the provided values
        self.request_type = request_type
        self.slo_constraint = slo_constraint
        self.client_id = client_id
        self.collection_id = collection_id
        self.deadline = deadline
        self.output_len = output_len
        self.input_len = input_len
        # Set the request weight: HIGH for LATENCY requests, LOW for others
        self.request_weight = RequestTypeWeight.HIGH if \
                    request_type == RequestType.LATENCY else RequestTypeWeight.LOW
        self.prediction_task = prediction_task
        self.real_output_len = output_len
        self.prompt = prompt
        self.output = output
        self.stage_id = stage_id
        self.request_id = request_id
        self.state = state

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]) -> "RequestInfo":
        '''
        Class method to create a RequestInfo instance from a JSON object.
        This method is useful for deserializing request data from a JSON structure.
        
        Args:
            json_obj (Dict[str, Any]): A dictionary containing request information.
            
        Returns:
            RequestInfo: A new instance of the RequestInfo class.
        '''
        # Extract values from the JSON object
        request_type = RequestType(json_obj["request_type"])  # Convert to RequestType enum
        slo_constraint = json_obj.get("slo_constraint", (0.0, 0.0, 0.0))  # Default to (0.0, 0.0, 0.0) if not provided
        client_id = json_obj.get("client_id", 0)  # Default to 0 if not provided
        collection_id = json_obj["collection_id"]
        deadline = json_obj["deadline"]
        output_len = json_obj["output_len"]
        input_len = json_obj.get("prompt_len", json_obj.get("input_len", 0))  # Support both field names
        prompt = json_obj.get("prompt", "")
        output = json_obj.get("output", "")
        stage_id = json_obj.get("stage_id", 0)
        request_id = json_obj.get("request_id", 0)
        state = json_obj.get("state", "")
        
        # Return a new instance of RequestInfo with the extracted values
        return cls(request_type, slo_constraint, client_id, collection_id, deadline, input_len, output_len,
                  None, prompt, output, stage_id, request_id, state)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert RequestInfo to dictionary format."""
        return {
            'prompt': self.prompt,
            'output': self.output,
            'prompt_len': self.input_len,
            'input_len': self.input_len,
            'output_len': self.output_len,
            'request_type': self.request_type.value,
            'collection_id': self.collection_id,
            'stage_id': self.stage_id,
            'request_id': self.request_id,
            'state': self.state,
            'deadline': self.deadline,
            'slo_constraint': self.slo_constraint,
            'client_id': self.client_id,
        }
    
class RequestConcordMetrics:
    def __init__(self, arrival_time) -> None:
        self.arrival_time: float = arrival_time
        self.prompt_len: int = 0
        self.output_len: int = 0
        self.TTFT: float = None
        self.TBT: List[float] = []
        self.TTLT: float = None
        self.new_prefill_tokens: int = 0
        self.new_decode_tokens: int = 0
        # should set to None when in swapped/waiting queue
        self.last_schedule_time: float = None
        self.service_gain: float = 0
        
    def reset(self) -> None:
        self.prompt_len = 0
        self.output_len = 0
        self.TTFT = None
        self.TBT = []
        self.TTLT = None
        self.new_prefill_tokens = 0
        self.new_decode_tokens = 0
        self.last_schedule_time = None
        self.service_gain = 0
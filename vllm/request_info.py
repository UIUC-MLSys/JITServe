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
    DECODE = 2
    
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
        slo_constraint = json_obj["slo_constraint"]
        client_id = json_obj["client_id"]
        collection_id = json_obj["collection_id"]
        deadline = json_obj["deadline"]
        output_len = json_obj["output_len"]
        input_len = json_obj["prompt_len"]
        
        # Return a new instance of RequestInfo with the extracted values
        return cls(request_type, slo_constraint, client_id, collection_id, deadline, input_len, output_len)
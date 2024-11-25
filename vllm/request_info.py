import asyncio
from enum import Enum, IntEnum
from typing import List, Dict, Any

# Enum class to define different types of requests
# LATENCY: Request that requires low latency
# THROUGHPUT: Request that requires high throughput
# COLLECTIVE: Request that involves collective operations
class RequestType(IntEnum):
    LATENCY = 0  # Latency-sensitive request
    THROUGHPUT = 1  # Throughput-sensitive request
    COLLECTIVE = 2  # Collective request (e.g., batch processing)

# Enum class to define the weight of a request
# HIGH: Request with high priority (used for latency-sensitive requests)
# LOW: Request with low priority (used for throughput-sensitive or collective requests)
class RequestTypeWeight(Enum):
    LOW = 1  # Low priority
    HIGH = 2  # High priority
    
class RequestPhaseWeight(Enum):
    PREFILL = 1
    DECODE = 2
    
def service_compute(prefilling_length: int, decoding_length: int) -> float:
    return prefilling_length * RequestPhaseWeight.PREFILL.value + \
        decoding_length * RequestPhaseWeight.DECODE.value

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
        client_id: int,  # ID of the client making the request
        collection_id: int,  # ID of the collection the request belongs to
        deadline: int,  # Deadline for the request to be processed
        output_len: int,  # The expected output length for the request
        prediction_task: asyncio.Task = None  # Task for prediction
    ):
        # Initialize the attributes with the provided values
        self.request_type = request_type
        self.client_id = client_id
        self.collection_id = collection_id
        self.deadline = deadline
        self.output_len = output_len
        # Set the request weight: HIGH for LATENCY requests, LOW for others
        self.request_weight = RequestTypeWeight.HIGH if \
                    request_type == RequestType.LATENCY else RequestTypeWeight.LOW
        self.prediction_task = prediction_task

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
        client_id = json_obj["client_id"]
        collection_id = json_obj["collection_id"]
        deadline = json_obj["deadline"]
        output_len = json_obj["output_len"]
        
        # Return a new instance of RequestInfo with the extracted values
        return cls(request_type, client_id, collection_id, deadline, output_len)
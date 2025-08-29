import math
import time
import threading
from itertools import permutations
from typing import List, Dict, Any, Optional, Tuple
from vllm.logger import init_logger

logger = init_logger("vllm")

Defalut_ToT_Stage = 4
Defalut_ToT_Requests_Pattern = [3, 1, 9, 1]

Default_DeepResearch_Stage = 6
Default_DeepResearch_Requests_Pattern = [1, 1, 1, 1, 1, 1]

# Define common graph structure classes

class Graph:
    def __init__(self, 
                nodes: List[List[tuple[int, Optional[int]]]],  # List of lists: each inner list represents nodes in one stage
                times: Optional[List[float]]=None,
                is_all_node: bool=False):
        self.nodes = nodes
        self.times = times
        self.is_all_node = is_all_node  # Flag for all-node approach
        
    def get_num_stages(self):
        """Get the number of stages in this graph."""
        return len(self.nodes)
    
    def get_stage_sizes(self):
        """Get the number of nodes in each stage."""
        return [len(stage_nodes) for stage_nodes in self.nodes]

    def __len__(self):
        return sum(self.get_stage_sizes())
    
class ToTStructure:
    def __init__(
        self, 
        stage_num: int = Defalut_ToT_Stage, 
        request_per_stage: List[int] = Defalut_ToT_Requests_Pattern,
        use_all_node: bool = False,
        stage_ratio_method: str = "execution_time"  # "execution_time" or "output_length"
    ) -> None:
        self.stage_num = stage_num
        self.request_per_stage = request_per_stage
        self.use_all_node = use_all_node
        self.stage_ratio_method = stage_ratio_method
        
        # Store individual request data for all-node approach
        self.stage_in_out_lengths: List[List[tuple]] = []  # List of (input_len, output_len) tuples per stage
        self.current_stage_requests: List[tuple] = []  # (input_len, output_len)
        
        self.current_time = time.time()
        self.stage_finish_time = []
        self.stage_finished = 0
        self.is_finished = False
        
        self._lock = threading.Lock()
        
        assert len(request_per_stage) == stage_num, "The length of request_per_stage should be equal to stage_num"
        
    def reset(self) -> None:
        self.stage_in_out_lengths = []
        self.current_stage_requests = []
        self.current_time = time.time()
        self.stage_finish_time = []
        self.stage_finished = 0
        self.is_finished = False
        
    def update(self) -> None:
        if (self.stage_finished) >= len(self.request_per_stage):
            return
        if len(self.current_stage_requests) == self.request_per_stage[self.stage_finished]:
            self.stage_finished += 1
            
            # Store the request data for this stage
            self.stage_in_out_lengths.append(self.current_stage_requests.copy())
            self.current_stage_requests = []
            
            stage_elapsed = time.time() - self.current_time
            # Ensure minimum stage time to avoid zero timing
            self.stage_finish_time.append(max(stage_elapsed, 0.001))
            self.current_time = time.time()
            
    def add_length(self, input_length: int, output_length: int) -> bool:
        """
        Add length information and check if the ToT structure is finished.
        
        Args:
            input_length: Input token length
            output_length: Output token length
            
        Returns:
            True if the ToT structure is finished, False otherwise
        """
        with self._lock:
            self.current_stage_requests.append((input_length, output_length))
            self.update()
        
            # Check if finished by stage completion
            if self.stage_finished == self.stage_num:
                self.is_finished = True
                return True
            else:
                return False
        
    def convert_to_graph(self) -> Graph:
        """Convert ToT structure to new graph format with configurable approaches."""
        nodes = []
        
        if self.use_all_node:
            # All-node approach: each request becomes a node
            for stage_idx in range(len(self.stage_in_out_lengths)):
                stage_nodes = []
                
                stage_requests = self.stage_in_out_lengths[stage_idx]
                
                # Create tuple for each request (input_len, output_len)
                for input_len, output_len in stage_requests:
                    stage_nodes.append((input_len, output_len))
                    
                nodes.append(stage_nodes)
        else:
            # Super-node approach: one node per stage
            for stage_idx in range(len(self.stage_in_out_lengths)):
                stage_requests = self.stage_in_out_lengths[stage_idx]
                
                # Sum up all requests in stage
                total_input = sum(req[0] for req in stage_requests)
                total_output = sum(req[1] for req in stage_requests)
                nodes.append([(total_input, total_output)])
        
        # Calculate stage ratios based on method
        if self.stage_ratio_method == "execution_time" and self.stage_finish_time:
            total_time = sum(self.stage_finish_time)
            if total_time <= 0:
                total_time = 0.001
            stage_ratios = [t / total_time for t in self.stage_finish_time]
        elif self.stage_ratio_method == "output_length" and self.stage_in_out_lengths:
            # Use max output length of each stage
            stage_max_outputs = [max(req[1] for req in stage_requests) if stage_requests else 0 
                               for stage_requests in self.stage_in_out_lengths]
            total_output = sum(stage_max_outputs)
            if total_output > 0:
                stage_ratios = [max_out / total_output for max_out in stage_max_outputs]
            else:
                stage_ratios = [1.0 / len(self.stage_in_out_lengths)] * len(self.stage_in_out_lengths)
        else:
            # Default uniform distribution
            num_stages = len(self.stage_in_out_lengths) if self.stage_in_out_lengths else self.stage_num
            stage_ratios = [1.0 / num_stages] * num_stages
        
        return Graph(nodes, stage_ratios, self.use_all_node)
    
    def convert_to_unfinished_graph(self, input_length: int, predict_output_length: int) -> Graph:
        """Convert unfinished ToT structure to graph with predicted next stage."""
        nodes = []
        
        # Add completed stages
        if self.use_all_node:
            # All-node approach
            for stage_idx in range(len(self.stage_in_out_lengths)):
                stage_nodes = []
                
                stage_requests = self.stage_in_out_lengths[stage_idx]
                
                for input_len, output_len in stage_requests:
                    stage_nodes.append((input_len, output_len))
                    
                nodes.append(stage_nodes)
            
            # Add predicted stage
            nodes.append([(input_length, predict_output_length)])
        else:
            # Super-node approach
            for stage_idx in range(len(self.stage_in_out_lengths)):
                stage_requests = self.stage_in_out_lengths[stage_idx]
                
                total_input = sum(req[0] for req in stage_requests)
                total_output = sum(req[1] for req in stage_requests)
                
                nodes.append([(total_input, total_output)])
            
            # Add predicted stage
            nodes.append([(input_length, predict_output_length)])
        
        return Graph(nodes, None, self.use_all_node)


class DeepResearchStructure:
    """
    Handles DeepResearch collective requests with flexible structure.
    Unlike ToT which has a fixed pattern, DeepResearch can have varying numbers
    of stages and requests per stage.
    """
    def __init__(self, use_all_node: bool = False, stage_ratio_method: str = "execution_time") -> None:
        self.collective_requests: Dict[int, Dict[str, Any]] = {}
        self.current_time = time.time()
        self.use_all_node = use_all_node
        self.stage_ratio_method = stage_ratio_method
        self._lock = threading.Lock()
        
    def reset(self) -> None:
        """Reset the structure for a new collective request."""
        with self._lock:
            self.collective_requests.clear()
            self.current_time = time.time()
    
    def initialize_collective(self, collective_id: int, num_stages: int, 
                            requests_per_stage: List[int]) -> None:
        """Initialize a new collective request structure."""
        with self._lock:
            self.collective_requests[collective_id] = {
                'num_stages': num_stages,
                'requests_per_stage': requests_per_stage,
                'completed_stages': 0,
                'stage_timings': [],
                'start_time': time.time(),
                'stage_start_time': time.time(),
                'is_finished': False,
                'current_stage_requests': [],
                # New storage for individual request data
                'stage_in_out_lengths': []  # List of (input_len, output_len) tuples per stage
            }
    
    def add_request_completion(self, collective_id: int, stage_id: int, 
                             input_length: int, output_length: int) -> bool:
        """
        Add a completed request to the collective and check if stage/collective is finished.
        
        Args:
            collective_id: ID of the collective request
            stage_id: Stage index within the collective
            input_length: Input token length
            output_length: Output token length
            
        Returns:
            True if the entire collective is finished, False otherwise
        """
        with self._lock:
            if collective_id not in self.collective_requests:
                logger.warning(f"Collective {collective_id} not initialized")
                return False
                
            collective = self.collective_requests[collective_id]
            
            # Check if collective is already finished
            if collective['is_finished']:
                return True
            
            # Check if we've completed all stages
            current_stage = collective['completed_stages']
            if current_stage >= collective['num_stages']:
                collective['is_finished'] = True
                return True
            
            # Add the request completion
            collective['current_stage_requests'].append((input_length, output_length))
            
            # Check if current stage is complete
            if current_stage < len(collective['requests_per_stage']):
                expected_requests = collective['requests_per_stage'][current_stage]
            else:
                # If stage_id exceeds our pattern, assume 1 request per stage
                expected_requests = 1
            
            if len(collective['current_stage_requests']) == expected_requests:
                # Stage completed, store individual request data
                collective['stage_in_out_lengths'].append(collective['current_stage_requests'].copy())
                collective['stage_timings'].append(time.time() - collective['stage_start_time'])
                collective['completed_stages'] += 1
                collective['current_stage_requests'] = []
                collective['stage_start_time'] = time.time()
                
                # Check if entire collective is finished
                if collective['completed_stages'] >= collective['num_stages']:
                    collective['is_finished'] = True
                    return True
                
            return False
    
    def is_collective_initialized(self, collective_id: int) -> bool:
        """Check if a collective request is initialized."""
        with self._lock:
            if collective_id not in self.collective_requests:
                return False
            return True
    
    def convert_to_graph(self, collective_id: int) -> Optional[Graph]:
        """Convert a completed collective request to a graph."""
        with self._lock:
            if collective_id not in self.collective_requests:
                return None
                
            collective = self.collective_requests[collective_id]
            
            if not collective['stage_in_out_lengths']:
                return None
                
            nodes = []
            
            if self.use_all_node:
                # All-node approach: each request becomes a node
                for stage_idx in range(len(collective['stage_in_out_lengths'])):
                    stage_nodes = []
                    
                    stage_requests = collective['stage_in_out_lengths'][stage_idx]
                    
                    # Create tuple for each request (input_len, output_len)
                    for input_len, output_len in stage_requests:
                        stage_nodes.append((input_len, output_len))
                        
                    nodes.append(stage_nodes)
            else:
                # Super-node approach: one node per stage
                for stage_idx in range(len(collective['stage_in_out_lengths'])):
                    stage_requests = collective['stage_in_out_lengths'][stage_idx]
                    
                    # Sum up all requests in stage
                    total_input = sum(req[0] for req in stage_requests)
                    total_output = sum(req[1] for req in stage_requests)
                    nodes.append([(total_input, total_output)])
            
            # Calculate stage ratios based on method
            if self.stage_ratio_method == "execution_time" and collective['stage_timings']:
                total_time = sum(collective['stage_timings'])
                if total_time > 0:
                    stage_ratios = [t / total_time for t in collective['stage_timings']]
                else:
                    stage_ratios = [1.0 / len(collective['stage_timings'])] * len(collective['stage_timings'])
            elif self.stage_ratio_method == "output_length" and collective['stage_in_out_lengths']:
                # Use max output length of each stage
                stage_max_outputs = [max(req[1] for req in stage_requests) if stage_requests else 0 
                                   for stage_requests in collective['stage_in_out_lengths']]
                total_output = sum(stage_max_outputs)
                if total_output > 0:
                    stage_ratios = [max_out / total_output for max_out in stage_max_outputs]
                else:
                    stage_ratios = [1.0 / len(collective['stage_in_out_lengths'])] * len(collective['stage_in_out_lengths'])
            else:
                # Default uniform distribution
                num_stages = len(collective['stage_in_out_lengths'])
                stage_ratios = [1.0 / num_stages] * num_stages

            # logger.info(f"nodes: {nodes}, stage_ratios: {stage_ratios}")

            return Graph(nodes, stage_ratios, self.use_all_node)
    
    def convert_to_unfinished_graph(self, collective_id: int, 
                                   input_length: int, 
                                   predict_output_length: int) -> Optional[Graph]:
        """Convert an unfinished collective request to a graph for prediction."""
        with self._lock:
            if collective_id not in self.collective_requests:
                return None
                
            collective = self.collective_requests[collective_id]
            nodes = []
            
            # Add completed stages
            if self.use_all_node:
                # All-node approach
                for stage_idx in range(len(collective['stage_in_out_lengths'])):
                    stage_nodes = []
                    
                    stage_requests = collective['stage_in_out_lengths'][stage_idx]
                    
                    for input_len, output_len in stage_requests:
                        stage_nodes.append((input_len, output_len))
                        
                    nodes.append(stage_nodes)
                
                # Add predicted stage
                nodes.append([(input_length, predict_output_length)])
            else:
                # Super-node approach
                for stage_idx in range(len(collective['stage_in_out_lengths'])):
                    stage_requests = collective['stage_in_out_lengths'][stage_idx]
                    
                    total_input = sum(req[0] for req in stage_requests)
                    total_output = sum(req[1] for req in stage_requests)
                    
                    nodes.append([(total_input, total_output)])
                
                # Add predicted stage
                nodes.append([(input_length, predict_output_length)])
            
            return Graph(nodes, None, self.use_all_node)


def predict_stage_ratio(query_graph: Graph, graph_set) -> float:
    stage = len(query_graph.nodes)
    best_graph: Graph | None = match_graph(query_graph, graph_set)

    if best_graph is None or best_graph.times is None:
        return sum(Defalut_ToT_Requests_Pattern[:stage]) / sum(Defalut_ToT_Requests_Pattern)
    else:
        if stage < len(best_graph.times):
            return sum(best_graph.times[:stage]) / sum(best_graph.times)
        else:
            return sum(Defalut_ToT_Requests_Pattern[:stage]) / sum(Defalut_ToT_Requests_Pattern)

    
def match_graph(query_graph, graph_set, input_w=0.3, output_w=0.7, sigma_input=1.0, sigma_output=1.0) -> Graph:
    best_graph = None
    best_similarity = -float('inf')

    for graph in graph_set:
        similarity = compute_similarity(query_graph, graph, input_w=input_w, output_w=output_w, 
                                       sigma_input=sigma_input, sigma_output=sigma_output)
        
        # update best graph
        if similarity > best_similarity:
            best_similarity = similarity
            best_graph = graph
    
    return best_graph

def compute_similarity(query_graph: Graph, target_graph: Graph, input_w=0.3, output_w=0.7, sigma_input=1.0, sigma_output=1.0):
    """
    Compute the similarity between a query graph and a target graph using new tuple-based node format.
    
    Args:
        query_graph: The query graph object, containing nodes as tuples (input_len, output_len).
        target_graph: The target graph object, containing nodes as tuples (input_len, output_len).
        input_w: The weight for input length similarity.
        output_w: The weight for output length similarity.
        sigma_input: The scale parameter for the Gaussian kernel used in input similarity.
        sigma_output: The scale parameter for the Gaussian kernel used in output similarity.

    Returns:
        A similarity score representing the similarity between the two graphs.
    """
    # Early return if target has fewer stages than query
    if target_graph.get_num_stages() < query_graph.get_num_stages():
        return 0.0
    
    total_similarity = 0.0
    
    # Compare stage by stage
    for stage_idx in range(query_graph.get_num_stages()):
        query_stage_nodes = query_graph.nodes[stage_idx].copy()
        target_stage_nodes = target_graph.nodes[stage_idx].copy()
        
        # Expand to match lengths
        if len(query_stage_nodes) > len(target_stage_nodes):
            # Expand target with (0, 0) nodes
            while len(target_stage_nodes) < len(query_stage_nodes):
                target_stage_nodes.append((0, 0))
        elif len(query_stage_nodes) < len(target_stage_nodes):
            # Expand query with appropriate nodes
            if query_stage_nodes and query_stage_nodes[0][1] is not None:
                # Expand with (0, 0) if output is not None
                expand_node = (0, 0)
            else:
                # Expand with (0, None) if output is None
                expand_node = (0, None)
            while len(query_stage_nodes) < len(target_stage_nodes):
                query_stage_nodes.append(expand_node)
        
        # Now both lists have same length, find best pairwise matching
        stage_similarity = find_best_matching_similarity(
            query_stage_nodes, target_stage_nodes, 
            input_w, output_w, sigma_input, sigma_output
        )
        total_similarity += stage_similarity
    
    return total_similarity


def find_best_matching_similarity(query_nodes, target_nodes, input_w, output_w, sigma_input, sigma_output):
    """
    Find the best pairwise matching between query and target nodes to maximize similarity.
    
    Args:
        query_nodes: List of query nodes (tuples)
        target_nodes: List of target nodes (tuples) 
        input_w, output_w: Weights for input and output similarity
        sigma_input, sigma_output: Scale parameters for Gaussian kernels
    
    Returns:
        Maximum similarity score for this matching
    """
    if not query_nodes or not target_nodes:
        return 0.0
        
    max_similarity = 0.0
    
    # Try all permutations of target nodes to find best matching
    for target_perm in permutations(target_nodes):
        similarity = 0.0
        for q_node, t_node in zip(query_nodes, target_perm):
            similarity += node_similarity(q_node, t_node, input_w, output_w, sigma_input, sigma_output)
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity


def node_similarity(node1, node2, input_w, output_w, sigma_input, sigma_output):
    """
    Compute similarity between two nodes using Gaussian kernels for input and output lengths.
    
    Args:
        node1: Tuple (input_len, output_len) or (input_len, None)
        node2: Tuple (input_len, output_len) or (input_len, None)
        input_w: Weight for input similarity
        output_w: Weight for output similarity
        sigma_input: Scale parameter for input Gaussian kernel
        sigma_output: Scale parameter for output Gaussian kernel
    
    Returns:
        Similarity score between the two nodes
    """
    input1, output1 = node1
    input2, output2 = node2
    
    # Input similarity using Gaussian kernel
    input_delta = (input1 - input2)
    input_sim = math.exp(-input_delta**2 / (2 * sigma_input**2))
    
    # Output similarity using Gaussian kernel
    if output1 is None or output2 is None:
        # If either output is None, output similarity is 0
        output_sim = 0.0
    else:
        output_delta = (output1 - output2)
        output_sim = math.exp(-output_delta**2 / (2 * sigma_output**2))
    
    # Weighted sum
    return input_w * input_sim + output_w * output_sim

def predict_deepresearch_stage_ratio(query_graph: Graph, graph_set, 
                                    stage_index: int = 0) -> float:
    """
    Predict stage ratio for DeepResearch collective requests.
    
    Args:
        query_graph: The query graph from an unfinished collective
        graph_set: Set of completed graphs to match against
        stage_index: The stage index for which to predict ratio
        
    Returns:
        Predicted stage ratio
    """
    best_graph = match_graph(query_graph, graph_set)
    
    if best_graph is None or best_graph.times is None or stage_index >= len(best_graph.times):
        # Default fallback: assume uniform distribution
        return sum(Default_DeepResearch_Requests_Pattern[:stage_index+1]) / sum(Default_DeepResearch_Requests_Pattern)
    else:
        logger.info(f"stage_index: {stage_index}, sum(best_graph.times[:stage_index+1]) / sum(best_graph.times): {sum(best_graph.times[:stage_index+1]) / sum(best_graph.times)}")
        # logger.info(f"best_graph.times: {best_graph.times}")
        # logger.info(f"best_graph.nodes: {best_graph.nodes}")
        return sum(best_graph.times[:stage_index+1]) / sum(best_graph.times)


def is_tot_request(request_type: int) -> bool:
    """Check if a request type corresponds to ToT (Tree of Thoughts)."""
    # Assuming ToT requests have a specific type identifier
    # This should be coordinated with the request type definitions
    return request_type in [0, 1]  # LATENCY or THROUGHPUT


def is_deepresearch_request(request_type: int) -> bool:
    """Check if a request type corresponds to DeepResearch collective."""
    return request_type == 2  # COLLECTIVE

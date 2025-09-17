import math
import time
import threading
import random
import asyncio
from itertools import permutations
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import copy
from vllm.logger import init_logger

logger = init_logger("vllm")

Default_ToT_Stage = 4
Default_ToT_Requests_Pattern = [3, 1, 3, 1]

Default_DeepResearch_Stage = 6
Default_DeepResearch_Requests_Pattern = [1, 1, 1, 1, 1, 1]

class RequestApplication(Enum):
    TOT = 1
    DEEPRESEARCH = 2

# Define common graph structure classes

class Graph:
    def __init__(self, 
                nodes: List[List[tuple[int, Optional[int]]]],  # List of lists: each inner list represents nodes in one stage
                times: Optional[List[float]]=None,
                is_all_node: bool=False,
                application: Optional[RequestApplication]=None):
        self.nodes = nodes
        self.times = times
        self.is_all_node = is_all_node  # Flag for all-node approach
        self.application = application
        
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
        stage_num: int = Default_ToT_Stage, 
        request_per_stage: List[int] = Default_ToT_Requests_Pattern,
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

    def add_length(self, input_length: int, output_length: Optional[int]) -> bool:
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
            # for stage_idx in range(len(self.stage_in_out_lengths)):
            #     stage_nodes = []
                
            #     stage_requests = self.stage_in_out_lengths[stage_idx]
                
            #     # Create tuple for each request (input_len, output_len)
            #     for input_len, output_len in stage_requests:
            #         stage_nodes.append((input_len, output_len))

            nodes = copy.deepcopy(self.stage_in_out_lengths)
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
        
        return Graph(nodes, stage_ratios, self.use_all_node, RequestApplication.TOT)

    def convert_to_unfinished_graph(self, input_length: int, predict_output_length: Optional[int]) -> Graph:
        """Convert unfinished ToT structure to graph with predicted next stage."""
        nodes = []
        
        # Add completed stages
        if self.use_all_node:
            # All-node approach
            # for stage_idx in range(len(self.stage_in_out_lengths)-1):
            #     stage_nodes = []
                
            #     stage_requests = self.stage_in_out_lengths[stage_idx]
                
            #     for input_len, output_len in stage_requests:
            #         stage_nodes.append((input_len, output_len))
                    
            #     nodes.append(stage_nodes)

            nodes = copy.deepcopy(self.stage_in_out_lengths)
            stage_idx = len(self.stage_in_out_lengths) - 1

            if stage_idx >= 0 and len(nodes[stage_idx]) < Default_ToT_Requests_Pattern[stage_idx]:
                nodes[stage_idx].append((input_length, predict_output_length))
            else:
                nodes.append([(input_length, predict_output_length)])

            # # Add predicted stage
            # nodes.append([(input_length, predict_output_length)])
        else:
            # Super-node approach
            for stage_idx in range(len(self.stage_in_out_lengths)):
                stage_requests = self.stage_in_out_lengths[stage_idx]
                
                total_input = sum(req[0] for req in stage_requests)
                total_output = sum(req[1] for req in stage_requests)
                
                nodes.append([(total_input, total_output)])
            
            # Add predicted stage
            nodes.append([(input_length, predict_output_length)])
        
        return Graph(nodes, None, self.use_all_node, RequestApplication.TOT)


class DeepResearchStructure:
    """
    Handles DeepResearch collective requests with flexible structure.
    Unlike ToT which has a fixed pattern, DeepResearch can have varying numbers
    of stages and requests per stage.
    """
    def __init__(self, 
                 num_stages: int = Default_DeepResearch_Stage,
                 requests_per_stage: List[int] = Default_DeepResearch_Requests_Pattern,
                 use_all_node: bool = False, 
                 stage_ratio_method: str = "execution_time") -> None:
        self.num_stages = num_stages
        self.requests_per_stage = requests_per_stage
        self.use_all_node = use_all_node
        self.stage_ratio_method = stage_ratio_method
        
        # Store individual request data for all-node approach
        self.stage_in_out_lengths: List[List[tuple]] = []  # List of (input_len, output_len) tuples per stage
        self.current_stage_requests: List[tuple] = []  # (input_len, output_len)
        
        self.current_time = time.time()
        self.stage_timings: List[float] = []
        self.completed_stages = 0
        self.is_finished = False
        
        self._lock = threading.Lock()
        
    def reset(self) -> None:
        """Reset the structure for a new collective request."""
        with self._lock:
            self.stage_in_out_lengths = []
            self.current_stage_requests = []
            self.stage_timings = []
            self.completed_stages = 0
            self.is_finished = False
            self.current_time = time.time()

    def add_length(self, input_length: int, output_length: Optional[int]) -> bool:
        """
        Add length information and check if the DeepResearch structure is finished.
        
        Args:
            input_length: Input token length
            output_length: Output token length
            
        Returns:
            True if the DeepResearch structure is finished, False otherwise
        """
        with self._lock:
            # Check if already finished
            if self.is_finished:
                return True
            
            # Add the request completion
            self.current_stage_requests.append((input_length, output_length))
            
            # Check if current stage is complete
            if self.completed_stages < len(self.requests_per_stage):
                expected_requests = self.requests_per_stage[self.completed_stages]
            else:
                # If stage exceeds our pattern, assume 1 request per stage
                expected_requests = 1
            
            if len(self.current_stage_requests) == expected_requests:
                # Stage completed, store individual request data
                self.stage_in_out_lengths.append(self.current_stage_requests.copy())
                stage_elapsed = time.time() - self.current_time
                # Ensure minimum stage time to avoid zero timing
                self.stage_timings.append(max(stage_elapsed, 0.001))
                self.completed_stages += 1
                self.current_stage_requests = []
                self.current_time = time.time()
                
                # Check if entire collective is finished
                if self.completed_stages >= self.num_stages:
                    self.is_finished = True
                    return True
                
            return False
    
    
    def convert_to_graph(self) -> Graph:
        """Convert DeepResearch structure to new graph format with configurable approaches."""
        nodes = []
        
        if self.use_all_node:
            # All-node approach: each request becomes a node
            # for stage_idx in range(len(self.stage_in_out_lengths)):
            #     stage_nodes = []
                
            #     stage_requests = self.stage_in_out_lengths[stage_idx]
                
            #     # Create tuple for each request (input_len, output_len)
            #     for input_len, output_len in stage_requests:
            #         stage_nodes.append((input_len, output_len))
                    
            #     nodes.append(stage_nodes)
            nodes = copy.deepcopy(self.stage_in_out_lengths)
        else:
            # Super-node approach: one node per stage
            for stage_idx in range(len(self.stage_in_out_lengths)):
                stage_requests = self.stage_in_out_lengths[stage_idx]
                
                # Sum up all requests in stage
                total_input = sum(req[0] for req in stage_requests)
                total_output = sum(req[1] for req in stage_requests)
                nodes.append([(total_input, total_output)])
        
        # Calculate stage ratios based on method
        if self.stage_ratio_method == "execution_time" and self.stage_timings:
            total_time = sum(self.stage_timings)
            if total_time <= 0:
                total_time = 0.001
            stage_ratios = [t / total_time for t in self.stage_timings]
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
            num_stages = len(self.stage_in_out_lengths) if self.stage_in_out_lengths else self.num_stages
            stage_ratios = [1.0 / num_stages] * num_stages
        
        return Graph(nodes, stage_ratios, self.use_all_node, RequestApplication.DEEPRESEARCH)

    def convert_to_unfinished_graph(self, input_length: int, predict_output_length: Optional[int]) -> Graph:
        """Convert unfinished DeepResearch structure to graph with predicted next stage."""
        nodes = []
        
        # Add completed stages
        if self.use_all_node:
            # All-node approach
            # for stage_idx in range(len(self.stage_in_out_lengths)):
            #     stage_nodes = []
                
            #     stage_requests = self.stage_in_out_lengths[stage_idx]
                
            #     for input_len, output_len in stage_requests:
            #         stage_nodes.append((input_len, output_len))
                    
            #     nodes.append(stage_nodes)
            nodes = copy.deepcopy(self.stage_in_out_lengths)
            stage_idx = len(self.stage_in_out_lengths) - 1

            if stage_idx >= 0 and len(nodes[stage_idx]) < self.requests_per_stage[stage_idx]:
                nodes[stage_idx].append((input_length, predict_output_length))
            else:
                nodes.append([(input_length, predict_output_length)])
            
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
        
        return Graph(nodes, None, self.use_all_node, RequestApplication.DEEPRESEARCH)


def predict_stage_ratio(query_graph: Graph, graph_set) -> float:
    stage = len(query_graph.nodes)
    best_graph: Graph | None = match_graph(query_graph, graph_set)

    if best_graph is None or best_graph.times is None:
        return sum(Default_ToT_Requests_Pattern[:stage]) / sum(Default_ToT_Requests_Pattern)
    else:
        if stage < len(best_graph.times):
            return sum(best_graph.times[:stage]) / sum(best_graph.times)
        else:
            return sum(Default_ToT_Requests_Pattern[:stage]) / sum(Default_ToT_Requests_Pattern)

    
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
    
    # Check if graphs are from different applications - return 0 similarity
    if (query_graph.application is not None and target_graph.application is not None and 
        query_graph.application != target_graph.application):
        return 0.0
    # Early return if target has fewer stages than query
    if target_graph.get_num_stages() < query_graph.get_num_stages():
        return 0.0
    
    total_similarity = 0.0
    
    # Compare stage by stage
    for stage_idx in range(query_graph.get_num_stages()):
        query_stage_nodes = query_graph.nodes[stage_idx].copy()
        target_stage_nodes = target_graph.nodes[stage_idx].copy()
        
        # Expand to match lengths
        if len(query_stage_nodes) != len(target_stage_nodes):
            return 0.0  # Require exact match in number of nodes per stage
        
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
        
    # For performance, limit permutation search for large node lists
    # Use greedy matching for lists larger than 6 nodes to avoid factorial explosion
    if len(target_nodes) > 6:
        return _greedy_matching_similarity(query_nodes, target_nodes, input_w, output_w, sigma_input, sigma_output)
        
    max_similarity = 0.0
    
    # Try all permutations of target nodes to find best matching
    for target_perm in permutations(target_nodes):
        similarity = 0.0
        for q_node, t_node in zip(query_nodes, target_perm):
            similarity += node_similarity(q_node, t_node, input_w, output_w, sigma_input, sigma_output)
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity


def _greedy_matching_similarity(query_nodes, target_nodes, input_w, output_w, sigma_input, sigma_output):
    """
    Fast greedy matching algorithm to avoid factorial time complexity.
    
    Args:
        query_nodes: List of query nodes (tuples)
        target_nodes: List of target nodes (tuples) 
        input_w, output_w: Weights for input and output similarity
        sigma_input, sigma_output: Scale parameters for Gaussian kernels
    
    Returns:
        Similarity score using greedy matching
    """
    if not query_nodes or not target_nodes:
        return 0.0
    
    used_targets = set()
    total_similarity = 0.0
    
    # For each query node, find the best available target node
    for q_node in query_nodes:
        best_similarity = 0.0
        best_target_idx = -1
        
        for i, t_node in enumerate(target_nodes):
            if i not in used_targets:
                similarity = node_similarity(q_node, t_node, input_w, output_w, sigma_input, sigma_output)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_target_idx = i
        
        if best_target_idx >= 0:
            used_targets.add(best_target_idx)
            total_similarity += best_similarity
    
    return total_similarity


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
        ratio_1 = sum(best_graph.times[:stage_index+1]) / sum(best_graph.times)
        ratio_2 = sum(Default_DeepResearch_Requests_Pattern[:stage_index+1]) / sum(Default_DeepResearch_Requests_Pattern)
        return max(ratio_1, ratio_2)


def graph_distance(query_graph: Graph, target_graph: Graph, input_w=0.3, output_w=0.7, sigma_input=1.0, sigma_output=1.0):
    """
    Compute the distance between a query graph and a target graph using new tuple-based node format.
    Returns 0.0 if two graphs have different number of stages, otherwise uses the same logic as compute_similarity.
    
    Args:
        query_graph: The query graph object, containing nodes as tuples (input_len, output_len).
        target_graph: The target graph object, containing nodes as tuples (input_len, output_len).
        input_w: The weight for input length similarity.
        output_w: The weight for output length similarity.
        sigma_input: The scale parameter for the Gaussian kernel used in input similarity.
        sigma_output: The scale parameter for the Gaussian kernel used in output similarity.

    Returns:
        A distance score representing the distance between the two graphs.
    """
    # Return 0.0 if two graphs have different number of stages
    if target_graph.get_num_stages() != query_graph.get_num_stages():
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


async def k_medoids_clustering(graphs: List[Graph], k: int, max_iterations: int = 100):
    """
    Perform k-medoids clustering on a list of graphs using graph_distance as the distance metric.
    
    Args:
        graphs: List of Graph objects to cluster
        k: Number of clusters
        max_iterations: Maximum number of iterations for the algorithm
    
    Returns:
        Tuple of (medoids_indices, cluster_assignments)
        - medoids_indices: List of indices of the k medoids in the original graphs list
        - cluster_assignments: List of cluster assignments for each graph (0 to k-1)
    """
    if not graphs or k <= 0 or k > len(graphs):
        return [], []
    
    n = len(graphs)
    
    # # For very large datasets, reduce max iterations to prevent timeouts
    # if n > 500:
    #     max_iterations = min(max_iterations, 20)
    # elif n > 200:
    #     max_iterations = min(max_iterations, 50)
    
    # Initialize medoids randomly
    medoids_indices = random.sample(range(n), k)
    
    for iteration in range(max_iterations):
        # Assign each graph to the nearest medoid
        cluster_assignments = []
        for i, graph in enumerate(graphs):
            best_cluster = 0
            best_distance = graph_distance(graph, graphs[medoids_indices[0]])
            
            for j in range(1, k):
                distance = graph_distance(graph, graphs[medoids_indices[j]])
                if distance > best_distance:  # Higher similarity means closer
                    best_distance = distance
                    best_cluster = j
            
            cluster_assignments.append(best_cluster)
            
            # Yield control every 10 graphs to prevent blocking
            if i % 10 == 0:
                await asyncio.sleep(0)
        
        # Update medoids
        new_medoids = []
        changed = False
        
        for cluster_id in range(k):
            # Find all graphs in this cluster
            cluster_graphs = [i for i, assignment in enumerate(cluster_assignments) if assignment == cluster_id]
            
            if not cluster_graphs:
                # If cluster is empty, keep the current medoid
                new_medoids.append(medoids_indices[cluster_id])
                continue
            
            # Find the graph that minimizes total distance to all graphs in the cluster
            # Use a more efficient O(n^2) approach
            best_medoid = cluster_graphs[0]
            best_total_distance = -float('inf')
            
            # For each potential medoid, calculate sum of distances to all other graphs
            for candidate_idx, candidate in enumerate(cluster_graphs):
                total_distance = 0
                for other_idx in cluster_graphs:
                    if candidate != other_idx:
                        total_distance += graph_distance(graphs[candidate], graphs[other_idx])
                
                if total_distance > best_total_distance:  # Higher similarity is better
                    best_total_distance = total_distance
                    best_medoid = candidate
                
                # Yield control every few candidates to prevent blocking
                if candidate_idx % 5 == 0:
                    await asyncio.sleep(0)
            
            new_medoids.append(best_medoid)
            if best_medoid != medoids_indices[cluster_id]:
                changed = True
        
        medoids_indices = new_medoids
        
        # If no medoids changed, we've converged
        if not changed:
            break
    
    return medoids_indices, cluster_assignments


class DynamicClustering:
    """
    Manages dynamic clustering of graphs based on the number of finished collective requests.
    """
    def __init__(self):
        self.finished_requests: List[Graph] = []
        self.clusters: List[List[int]] = []  # List of clusters, each containing indices into finished_requests
        self.medoids: List[int] = []  # Indices of medoids in finished_requests
        self.cluster_thresholds = [100, 200, 300, 400, 500]  # More aggressive thresholds to reduce re-clustering frequency
        self.current_k = 0
        self._lock = threading.Lock()
        self._clustering_in_progress = False
    
    def add_finished_request(self, graph: Graph):
        """Add a finished collective request graph."""
        with self._lock:
            self.finished_requests.append(graph)
            # Schedule async clustering in background if we have an event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._update_clustering_async())
            except RuntimeError:
                # No event loop running, skip async clustering
                logger.warning("No event loop running, skipping async clustering update")
    
    async def _update_clustering_async(self):
        """Update clustering based on the current number of finished requests."""
        # Use a lock to prevent concurrent clustering operations
        if hasattr(self, '_clustering_in_progress') and self._clustering_in_progress:
            return
        
        self._clustering_in_progress = True
        
        try:
            with self._lock:
                n_requests = len(self.finished_requests)
                
                # Determine the target number of clusters
                target_k = 0
                for i, threshold in enumerate(self.cluster_thresholds):
                    if n_requests >= threshold:
                        target_k = i + 2  # Start with 2 clusters at 100 requests
                    else:
                        break
                
                # If we haven't reached the first threshold (100), no clustering needed
                if target_k == 0:
                    self.clusters = []
                    self.medoids = []
                    self.current_k = 0
                    return
                
                # If target k changed or we need to re-cluster
                should_recluster = (target_k != self.current_k) or (n_requests in self.cluster_thresholds)
                
            if should_recluster:
                logger.info(f"Re-clustering {n_requests} requests into {target_k} clusters")
                try:
                    # Use asyncio timeout instead of signal-based timeout
                    medoid_indices, assignments = await asyncio.wait_for(
                        k_medoids_clustering(self.finished_requests, target_k), 
                        timeout=30.0
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"Clustering operation failed or timed out: {e}, falling back to simple assignment")
                    # Fall back to simple random assignment
                    medoid_indices = random.sample(range(n_requests), min(target_k, n_requests))
                    assignments = [i % target_k for i in range(n_requests)]
                
                # Update clusters
                with self._lock:
                    self.clusters = [[] for _ in range(target_k)]
                    for i, cluster_id in enumerate(assignments):
                        self.clusters[cluster_id].append(i)
                    
                    self.medoids = medoid_indices
                    self.current_k = target_k
            else:
                # Just add the new request to the closest cluster
                with self._lock:
                    if self.medoids:
                        new_graph = self.finished_requests[-1]
                        best_cluster = 0
                        best_distance = graph_distance(new_graph, self.finished_requests[self.medoids[0]])
                        
                        for i in range(1, len(self.medoids)):
                            distance = graph_distance(new_graph, self.finished_requests[self.medoids[i]])
                            if distance > best_distance:
                                best_distance = distance
                                best_cluster = i
                        
                        self.clusters[best_cluster].append(len(self.finished_requests) - 1)
        finally:
            self._clustering_in_progress = False
    
    def get_cluster_info(self):
        """Get current clustering information."""
        with self._lock:
            return {
                'n_requests': len(self.finished_requests),
                'n_clusters': self.current_k,
                'cluster_sizes': [len(cluster) for cluster in self.clusters],
                'medoids': self.medoids.copy()
            }
    
    def find_best_match(self, query_graph: Graph) -> Optional[Graph]:
        """Find the best matching graph by first finding closest cluster, then most similar graph within that cluster."""
        with self._lock:
            if not self.medoids or not self.clusters:
                return None
            
            # Step 1: Find the closest cluster by comparing with medoids
            best_cluster_idx = 0
            best_cluster_similarity = -float('inf')
            
            for cluster_idx, medoid_idx in enumerate(self.medoids):
                medoid_graph = self.finished_requests[medoid_idx]
                similarity = graph_distance(query_graph, medoid_graph)
                if similarity > best_cluster_similarity:
                    best_cluster_similarity = similarity
                    best_cluster_idx = cluster_idx
            
            # Step 2: Find the most similar graph within the closest cluster
            closest_cluster = self.clusters[best_cluster_idx]
            if not closest_cluster:
                return None
                
            best_graph = None
            best_similarity = -float('inf')
            
            for graph_idx in closest_cluster:
                candidate_graph = self.finished_requests[graph_idx]
                similarity = graph_distance(query_graph, candidate_graph)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_graph = candidate_graph
            
            return best_graph

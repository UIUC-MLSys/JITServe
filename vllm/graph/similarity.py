import math
import time
import threading
from typing import List, Dict, Any, Optional
from vllm.logger import init_logger

logger = init_logger(__file__)

Defalut_ToT_Stage = 4
Defalut_ToT_Requests_Pattern = [3, 1, 9, 1]

Default_DeepResearch_Stage = 6
Default_DeepResearch_Requests_Pattern = [1, 1, 1, 1, 1, 1]

# Define common graph structure classes
class Vertex:
    def __init__(self, weight):
        self.weight = weight

class Edge:
    def __init__(self, weight):
        self.weight = weight

class Graph:
    def __init__(self, 
                vertices: List[List[Vertex]],  # List of lists: each inner list represents vertices in one stage
                edges: List[List[Edge]],       # List of lists: each inner list represents edges in one stage
                times: Optional[List[float]]=None,
                is_all_node: bool=False):
        self.vertices = vertices  # List of lists of vertices per stage
        self.edges = edges        # List of lists of edges per stage
        self.times = times
        self.is_all_node = is_all_node  # Flag for all-node approach
        
    def get_num_stages(self):
        """Get the number of stages in this graph."""
        return len(self.vertices)
    
    def get_stage_sizes(self):
        """Get the number of vertices in each stage."""
        return [len(stage_vertices) for stage_vertices in self.vertices]

    def __len__(self):
        return len(self.edges)
    
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
        self.stage_input_lengths: List[List[int]] = []
        self.stage_output_lengths: List[List[int]] = []
        self.current_stage_requests: List[Tuple[int, int]] = []  # (input_len, output_len)
        
        self.current_time = time.time()
        self.stage_finish_time = []
        self.stage_finished = 0
        self.is_finished = False
        
        self._lock = threading.Lock()
        
        assert len(request_per_stage) == stage_num, "The length of request_per_stage should be equal to stage_num"
        
    def reset(self) -> None:
        self.stage_input_lengths = []
        self.stage_output_lengths = []
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
            stage_inputs = [req[0] for req in self.current_stage_requests]
            stage_outputs = [req[1] for req in self.current_stage_requests]
            
            self.stage_input_lengths.append(stage_inputs)
            self.stage_output_lengths.append(stage_outputs)
            self.current_stage_requests = []
            
            stage_elapsed = time.time() - self.current_time
            # Ensure minimum stage time to avoid zero timing
            self.stage_finish_time.append(max(stage_elapsed, 0.001))
            self.current_time = time.time()
            
    def add_length(self, input_length: int, output_length: int, 
                   request_output_finished: bool = False) -> bool:
        """
        Add length information and check if the ToT structure is finished.
        
        Args:
            input_length: Input token length
            output_length: Output token length  
            request_output_finished: Whether request_output.finished is True
            
        Returns:
            True if the ToT structure is finished, False otherwise
        """
        with self._lock:
            self.current_stage_requests.append((input_length, output_length))
            self.update()
        
            # Check if finished by stage completion or request_output.finished flag
            if self.stage_finished == self.stage_num or request_output_finished:
                self.is_finished = True
                return True
            else:
                return False
        
    def convert_to_graph(self) -> Graph:
        """Convert ToT structure to new graph format with configurable approaches."""
        vertices = []
        edges = []
        
        if self.use_all_node:
            # All-node approach: each request becomes a node
            for stage_idx in range(len(self.stage_output_lengths)):
                stage_vertices = []
                stage_edges = []
                
                stage_outputs = self.stage_output_lengths[stage_idx]
                stage_inputs = self.stage_input_lengths[stage_idx]
                
                # Create vertex for each request (weight = output length)
                for output_len in stage_outputs:
                    stage_vertices.append(Vertex(output_len))
                
                # Create edge for each request (weight = input length)
                for input_len in stage_inputs:
                    stage_edges.append(Edge(input_len))
                    
                vertices.append(stage_vertices)
                edges.append(stage_edges)
        else:
            # Super-node approach: one node per stage
            for stage_idx in range(len(self.stage_output_lengths)):
                stage_outputs = self.stage_output_lengths[stage_idx]
                stage_inputs = self.stage_input_lengths[stage_idx]
                
                # Vertex weight = sum of output lengths in stage
                total_output = sum(stage_outputs)
                vertices.append([Vertex(total_output)])
                
                # Edge weight = sum of input lengths in stage
                total_input = sum(stage_inputs)
                edges.append([Edge(total_input)])
        
        # Calculate stage ratios based on method
        if self.stage_ratio_method == "execution_time" and self.stage_finish_time:
            total_time = sum(self.stage_finish_time)
            if total_time <= 0:
                total_time = 0.001
            stage_ratios = [t / total_time for t in self.stage_finish_time]
        elif self.stage_ratio_method == "output_length" and self.stage_output_lengths:
            # Use max output length of each stage
            stage_max_outputs = [max(stage_outputs) if stage_outputs else 0 
                               for stage_outputs in self.stage_output_lengths]
            total_output = sum(stage_max_outputs)
            if total_output > 0:
                stage_ratios = [max_out / total_output for max_out in stage_max_outputs]
            else:
                stage_ratios = [1.0 / len(self.stage_output_lengths)] * len(self.stage_output_lengths)
        else:
            # Default uniform distribution
            num_stages = len(self.stage_output_lengths) if self.stage_output_lengths else self.stage_num
            stage_ratios = [1.0 / num_stages] * num_stages
        
        return Graph(vertices, edges, stage_ratios, self.use_all_node)
    
    def convert_to_unfinished_graph(self, input_length: int, predict_output_length: int) -> Graph:
        """Convert unfinished ToT structure to graph with predicted next stage."""
        vertices = []
        edges = []
        
        # Add completed stages
        if self.use_all_node:
            # All-node approach
            for stage_idx in range(len(self.stage_output_lengths)):
                stage_vertices = []
                stage_edges = []
                
                stage_outputs = self.stage_output_lengths[stage_idx]
                stage_inputs = self.stage_input_lengths[stage_idx]
                
                for output_len in stage_outputs:
                    stage_vertices.append(Vertex(output_len))
                for input_len in stage_inputs:
                    stage_edges.append(Edge(input_len))
                    
                vertices.append(stage_vertices)
                edges.append(stage_edges)
            
            # Add predicted stage
            vertices.append([Vertex(predict_output_length)])
            edges.append([Edge(input_length)])
        else:
            # Super-node approach
            for stage_idx in range(len(self.stage_output_lengths)):
                stage_outputs = self.stage_output_lengths[stage_idx]
                stage_inputs = self.stage_input_lengths[stage_idx]
                
                total_output = sum(stage_outputs)
                total_input = sum(stage_inputs)
                
                vertices.append([Vertex(total_output)])
                edges.append([Edge(total_input)])
            
            # Add predicted stage
            vertices.append([Vertex(predict_output_length)])
            edges.append([Edge(input_length)])
        
        return Graph(vertices, edges, None, self.use_all_node)


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
                'stage_input_lengths': [],  # List of lists: each stage's input lengths
                'stage_output_lengths': []  # List of lists: each stage's output lengths
            }
    
    def add_request_completion(self, collective_id: int, stage_id: int, 
                             input_length: int, output_length: int, 
                             request_output_finished: bool = False) -> bool:
        """
        Add a completed request to the collective and check if stage/collective is finished.
        
        Args:
            collective_id: ID of the collective request
            stage_id: Stage index within the collective
            input_length: Input token length
            output_length: Output token length
            request_output_finished: Whether the request_output indicates finished status
            
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
                stage_inputs = [req[0] for req in collective['current_stage_requests']]
                stage_outputs = [req[1] for req in collective['current_stage_requests']]
                
                collective['stage_input_lengths'].append(stage_inputs)
                collective['stage_output_lengths'].append(stage_outputs)
                collective['stage_timings'].append(time.time() - collective['stage_start_time'])
                collective['completed_stages'] += 1
                collective['current_stage_requests'] = []
                collective['stage_start_time'] = time.time()
                
                # Check if entire collective is finished
                if (collective['completed_stages'] >= collective['num_stages'] or 
                    request_output_finished):
                    collective['is_finished'] = True
                    return True
            
            # Also check request_output.finished flag for early completion
            if request_output_finished:
                collective['is_finished'] = True
                return True
                
            return False
    
    def is_collective_finished(self, collective_id: int) -> bool:
        """Check if a collective request is finished."""
        with self._lock:
            if collective_id not in self.collective_requests:
                return False
            return self.collective_requests[collective_id]['is_finished']
    
    def convert_to_graph(self, collective_id: int) -> Optional[Graph]:
        """Convert a completed collective request to a graph."""
        with self._lock:
            if collective_id not in self.collective_requests:
                return None
                
            collective = self.collective_requests[collective_id]
            
            if not collective['stage_input_lengths'] or not collective['stage_output_lengths']:
                return None
                
            vertices = []
            edges = []
            
            if self.use_all_node:
                # All-node approach: each request becomes a node
                for stage_idx in range(len(collective['stage_output_lengths'])):
                    stage_vertices = []
                    stage_edges = []
                    
                    stage_outputs = collective['stage_output_lengths'][stage_idx]
                    stage_inputs = collective['stage_input_lengths'][stage_idx]
                    
                    # Create vertex for each request (weight = output length)
                    for output_len in stage_outputs:
                        stage_vertices.append(Vertex(output_len))
                    
                    # Create edge for each request (weight = input length)
                    for input_len in stage_inputs:
                        stage_edges.append(Edge(input_len))
                        
                    vertices.append(stage_vertices)
                    edges.append(stage_edges)
            else:
                # Super-node approach: one node per stage
                for stage_idx in range(len(collective['stage_output_lengths'])):
                    stage_outputs = collective['stage_output_lengths'][stage_idx]
                    stage_inputs = collective['stage_input_lengths'][stage_idx]
                    
                    # Vertex weight = sum of output lengths in stage
                    total_output = sum(stage_outputs)
                    vertices.append([Vertex(total_output)])
                    
                    # Edge weight = sum of input lengths in stage
                    total_input = sum(stage_inputs)
                    edges.append([Edge(total_input)])
            
            # Calculate stage ratios based on method
            if self.stage_ratio_method == "execution_time" and collective['stage_timings']:
                total_time = sum(collective['stage_timings'])
                if total_time > 0:
                    stage_ratios = [t / total_time for t in collective['stage_timings']]
                else:
                    stage_ratios = [1.0 / len(collective['stage_timings'])] * len(collective['stage_timings'])
            elif self.stage_ratio_method == "output_length" and collective['stage_output_lengths']:
                # Use max output length of each stage
                stage_max_outputs = [max(stage_outputs) if stage_outputs else 0 
                                   for stage_outputs in collective['stage_output_lengths']]
                total_output = sum(stage_max_outputs)
                if total_output > 0:
                    stage_ratios = [max_out / total_output for max_out in stage_max_outputs]
                else:
                    stage_ratios = [1.0 / len(collective['stage_output_lengths'])] * len(collective['stage_output_lengths'])
            else:
                # Default uniform distribution
                num_stages = len(collective['stage_output_lengths'])
                stage_ratios = [1.0 / num_stages] * num_stages
            
            return Graph(vertices, edges, stage_ratios, self.use_all_node)
    
    def convert_to_unfinished_graph(self, collective_id: int, 
                                   input_length: int, 
                                   predict_output_length: int) -> Optional[Graph]:
        """Convert an unfinished collective request to a graph for prediction."""
        with self._lock:
            if collective_id not in self.collective_requests:
                return None
                
            collective = self.collective_requests[collective_id]
            vertices = []
            edges = []
            
            # Add completed stages
            if self.use_all_node:
                # All-node approach
                for stage_idx in range(len(collective['stage_output_lengths'])):
                    stage_vertices = []
                    stage_edges = []
                    
                    stage_outputs = collective['stage_output_lengths'][stage_idx]
                    stage_inputs = collective['stage_input_lengths'][stage_idx]
                    
                    for output_len in stage_outputs:
                        stage_vertices.append(Vertex(output_len))
                    for input_len in stage_inputs:
                        stage_edges.append(Edge(input_len))
                        
                    vertices.append(stage_vertices)
                    edges.append(stage_edges)
                
                # Add predicted stage
                vertices.append([Vertex(predict_output_length)])
                edges.append([Edge(input_length)])
            else:
                # Super-node approach
                for stage_idx in range(len(collective['stage_output_lengths'])):
                    stage_outputs = collective['stage_output_lengths'][stage_idx]
                    stage_inputs = collective['stage_input_lengths'][stage_idx]
                    
                    total_output = sum(stage_outputs)
                    total_input = sum(stage_inputs)
                    
                    vertices.append([Vertex(total_output)])
                    edges.append([Edge(total_input)])
                
                # Add predicted stage
                vertices.append([Vertex(predict_output_length)])
                edges.append([Edge(input_length)])
            
            return Graph(vertices, edges, None, self.use_all_node)


def predict_stage_ratio(query_graph: Graph, graph_set) -> float:
    stage = len(query_graph.vertices) - 1
    best_graph: Graph | None = match_graph(query_graph, graph_set)
    
    if best_graph is None or best_graph.times is None:
        return Defalut_ToT_Requests_Pattern[stage] / sum(Defalut_ToT_Requests_Pattern)
    else:
        if stage < len(best_graph.times):
            return best_graph.times[stage]
        else:
            return 1.0 / len(best_graph.vertices)

    
def match_graph(query_graph, graph_set, alpha=0.5, beta=0.5, sigma_vertex=1.0, sigma_edge=1.0, graph_match=False) -> Graph:
    best_graph = None
    best_similarity = -float('inf')

    for graph in graph_set:
        similarity = compute_similarity(query_graph, graph, alpha=alpha, beta=beta, 
                                       sigma_vertex=sigma_vertex, sigma_edge=sigma_edge, 
                                       graph_match=graph_match)
        
        # update best graph
        if similarity > best_similarity:
            best_similarity = similarity
            best_graph = graph
    
    return best_graph

def compute_similarity(query_graph: Graph, target_graph: Graph, alpha=0.5, beta=0.5, sigma_vertex=1.0, sigma_edge=1.0, graph_match=False):
    """
    Compute the similarity between a query graph and a target graph.
    
    Args:
        query_graph: The query graph object, containing vertices and edges.
        target_graph: The target graph object, containing vertices and edges.
        alpha: The weight for vertex similarity.
        beta: The weight for edge similarity.
        sigma_vertex: The scale parameter for the Gaussian kernel used in vertex similarity.
        sigma_edge: The scale parameter for the Gaussian kernel used in edge similarity.

    Returns:
        A similarity score representing the similarity between the two graphs.
    """
    # Initialize similarity scores for vertices and edges
    vertex_score = 0
    edge_score = 0
    
    # Handle the new list-of-lists structure
    query_vertices = []
    query_edges = []
    target_vertices = []
    target_edges = []
    
    # Flatten vertices and edges for comparison
    for stage_vertices in query_graph.vertices:
        query_vertices.extend(stage_vertices)
    for stage_edges in query_graph.edges:
        query_edges.extend(stage_edges)
    for stage_vertices in target_graph.vertices:
        target_vertices.extend(stage_vertices)
    for stage_edges in target_graph.edges:
        target_edges.extend(stage_edges)

    # Match the graphs where DAG is undertermined
    if graph_match:
        # Compute vertex similarity
        for v_q in query_vertices:
            max_sim = 0
            for v_g in target_vertices:
                sim = vertex_similarity(v_q, v_g, sigma_vertex)
                max_sim = max(max_sim, sim)  # Select the maximum similarity for the vertex
            vertex_score += max_sim

        # Compute edge similarity
        for e_q in query_edges:
            max_sim = 0
            for e_g in target_edges:
                sim = edge_similarity(e_q, e_g, sigma_edge)
                max_sim = max(max_sim, sim)  # Select the maximum similarity for the edge
            edge_score += max_sim
    else:
        # Stage-wise comparison when not using graph matching
        min_stages = min(len(query_graph.vertices), len(target_graph.vertices))
        
        for stage_idx in range(min_stages):
            q_stage_vertices = query_graph.vertices[stage_idx]
            t_stage_vertices = target_graph.vertices[stage_idx]
            q_stage_edges = query_graph.edges[stage_idx]
            t_stage_edges = target_graph.edges[stage_idx]
            
            # Compare vertices in this stage
            min_vertices = min(len(q_stage_vertices), len(t_stage_vertices))
            for v_idx in range(min_vertices):
                sim = vertex_similarity(q_stage_vertices[v_idx], t_stage_vertices[v_idx], sigma_vertex)
                vertex_score += sim
            
            # Compare edges in this stage
            min_edges = min(len(q_stage_edges), len(t_stage_edges))
            for e_idx in range(min_edges):
                sim = edge_similarity(q_stage_edges[e_idx], t_stage_edges[e_idx], sigma_edge)
                edge_score += sim

    # Normalize scores by the number of vertices and edges in the query graph
    if query_vertices:
        vertex_score /= len(query_vertices)
    if query_edges:
        edge_score /= len(query_edges)

    # Return the weighted combined similarity
    return alpha * vertex_score + beta * edge_score


def vertex_similarity(v1, v2, sigma):
    """
    Compute similarity between two vertices using a Gaussian kernel.
    
    Args:
        v1: The first vertex, expected to have a 'weight' attribute.
        v2: The second vertex, expected to have a 'weight' attribute.
        sigma: The scale parameter for the Gaussian kernel.

    Returns:
        A similarity score between the two vertices.
    """
    if not hasattr(v1, 'weight') or not hasattr(v2, 'weight'):
        raise ValueError("Vertices must have 'weight' attribute for comparison.")

    # Compute the absolute difference in vertex weights (output lengths)
    delta = abs(v1.weight - v2.weight)
    # Apply Gaussian kernel to convert difference into similarity
    return math.exp(-delta**2 / (2 * sigma**2))


def edge_similarity(e1, e2, sigma):
    """
    Compute similarity between two edges using a Gaussian kernel.
    
    Args:
        e1: The first edge, expected to have a 'weight' attribute.
        e2: The second edge, expected to have a 'weight' attribute.
        sigma: The scale parameter for the Gaussian kernel.

    Returns:
        A similarity score between the two edges.
    """
    if not hasattr(e1, 'weight') or not hasattr(e2, 'weight'):
        raise ValueError("Edges must have 'weight' attribute for comparison.")

    # Compute the absolute difference in edge weights
    delta = abs(e1.weight - e2.weight)
    # Apply Gaussian kernel to convert difference into similarity
    return math.exp(-delta**2 / (2 * sigma**2))


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
    best_graph = match_graph(query_graph, graph_set, graph_match=True)
    
    if best_graph is None or best_graph.times is None or stage_index >= len(best_graph.times):
        # Default fallback: assume uniform distribution
        return 1.0 / max(1, len(query_graph.vertices))
    else:
        return best_graph.times[stage_index]


def is_tot_request(request_type: int) -> bool:
    """Check if a request type corresponds to ToT (Tree of Thoughts)."""
    # Assuming ToT requests have a specific type identifier
    # This should be coordinated with the request type definitions
    return request_type in [0, 1]  # LATENCY or THROUGHPUT


def is_deepresearch_request(request_type: int) -> bool:
    """Check if a request type corresponds to DeepResearch collective."""
    return request_type == 2  # COLLECTIVE

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

class Vertex:
    def __init__(self, attr):
        self.attr = attr

class Edge:
    def __init__(self, weight):
        self.weight = weight

class Graph:
    def __init__(self, vertices, edges, total_time, stage_ratio):
        self.vertices = vertices
        self.edges = edges
        self.total_time = total_time
        self.stage_ratio = stage_ratio

    def __len__(self):
        return len(self.edges)
    
class ToTStructure:
    def __init__(
        self, 
        stage_num: int = Defalut_ToT_Stage, 
        request_per_stage: List[int] = Defalut_ToT_Requests_Pattern
    ) -> None:
        self.stage_num = stage_num
        self.request_per_stage = request_per_stage
        self.output_input_ratio: List[float] = []
        self.current_stage_length: List[float] = []
        self.current_time = time.time()
        self.stage_finish_time = []
        self.stage_finished = 0
        self.is_finished = False
        
        self._lock = threading.Lock()
        
        assert len(request_per_stage) == stage_num, "The length of request_per_stage should be equal to stage_num"
        
    def reset(self) -> None:
        self.output_input_ratio = []
        self.current_stage_length = []
        self.current_time = time.time()
        self.stage_finish_time = []
        self.stage_finished = 0
        self.is_finished = False
        
    def update(self) -> None:
        if (self.stage_finished) >= len(self.request_per_stage):
            return
        if len(self.current_stage_length) == self.request_per_stage[self.stage_finished]:
            self.stage_finished += 1
            stage_input_length = sum([length[0] for length in self.current_stage_length])
            stage_output_length = sum([length[1] for length in self.current_stage_length])
            current_stage_ratio = stage_output_length / stage_input_length
            self.output_input_ratio.append(current_stage_ratio)
            self.current_stage_length = []
            
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
            self.current_stage_length.append((input_length, output_length))
            self.update()
        
            # Check if finished by stage completion or request_output.finished flag
            if self.stage_finished == self.stage_num or request_output_finished:
                self.is_finished = True
                return True
            else:
                return False
        
    def convert_to_graph(self) -> Graph:
        vertices = [Vertex(1) for _ in range(self.stage_num)]
        edges = [Edge(weight) for weight in self.output_input_ratio]
        total_time = sum(self.stage_finish_time) if self.stage_finish_time else 0.001
        
        # Ensure minimum total_time to avoid division by zero
        if total_time <= 0:
            total_time = 0.001  # 1ms minimum
            
        # Calculate stage ratios, handling empty stage_finish_time
        if self.stage_finish_time:
            stage_ratio = [stage_time / total_time for stage_time in self.stage_finish_time]
        else:
            # Default uniform distribution if no timing data
            stage_ratio = [1.0 / len(vertices)] * len(vertices)
        
        return Graph(vertices, edges, total_time, stage_ratio)
    
    def convert_to_unfinished_graph(self, input_length: int, predict_output_length: int) -> Graph:
        new_vertice = Vertex(1)
        assert input_length != 0, "The input_length should not be zero"
        new_edge = Edge(predict_output_length / input_length)
        vertices = []
        edges = []
        
        for i in range(self.stage_finished - 1):
            vertices.append(Vertex(1))
            edges.append(Edge(self.output_input_ratio[i]))
            
        vertices.append(new_vertice)
        edges.append(new_edge)
        
        return Graph(vertices, edges, 0, [])


class DeepResearchStructure:
    """
    Handles DeepResearch collective requests with flexible structure.
    Unlike ToT which has a fixed pattern, DeepResearch can have varying numbers
    of stages and requests per stage.
    """
    def __init__(self) -> None:
        self.collective_requests: Dict[int, Dict[str, Any]] = {}
        self.current_time = time.time()
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
                'stage_outputs': [],
                'stage_timings': [],
                'output_input_ratios': [],
                'start_time': time.time(),
                'stage_start_time': time.time(),
                'is_finished': False,
                'current_stage_requests': []
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
                # Stage completed, calculate metrics
                stage_input = sum(req[0] for req in collective['current_stage_requests'])
                stage_output = sum(req[1] for req in collective['current_stage_requests'])
                
                if stage_input > 0:
                    ratio = stage_output / stage_input
                else:
                    ratio = 1.0
                    
                collective['output_input_ratios'].append(ratio)
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
            
            if not collective['output_input_ratios']:
                return None
                
            # Create vertices for each completed stage
            vertices = [Vertex(1) for _ in range(len(collective['output_input_ratios']))]
            
            # Create edges with output/input ratios
            edges = [Edge(ratio) for ratio in collective['output_input_ratios']]
            
            # Calculate total time and stage ratios
            total_time = sum(collective['stage_timings'])
            if total_time > 0:
                stage_ratios = [t / total_time for t in collective['stage_timings']]
            else:
                stage_ratios = [1.0 / len(collective['stage_timings'])] * len(collective['stage_timings'])
            
            return Graph(vertices, edges, total_time, stage_ratios)
    
    def convert_to_unfinished_graph(self, collective_id: int, 
                                   input_length: int, 
                                   predict_output_length: int) -> Optional[Graph]:
        """Convert an unfinished collective request to a graph for prediction."""
        with self._lock:
            if collective_id not in self.collective_requests:
                return None
                
            collective = self.collective_requests[collective_id]
            
            # Create vertices and edges for completed stages
            vertices = []
            edges = []
            
            for i in range(len(collective['output_input_ratios'])):
                vertices.append(Vertex(1))
                edges.append(Edge(collective['output_input_ratios'][i]))
            
            # Add predicted stage
            if input_length > 0:
                vertices.append(Vertex(1))
                edges.append(Edge(predict_output_length / input_length))
            
            return Graph(vertices, edges, 0, [])


def predict_stage_ratio(query_graph: Graph, graph_set) -> float:
    stage = len(query_graph.vertices) - 1
    best_graph: Graph | None = match_graph(query_graph, graph_set)
    
    if best_graph is None:
        return Defalut_ToT_Requests_Pattern[stage] / sum(Defalut_ToT_Requests_Pattern)
    else:
        return best_graph.stage_ratio[stage]

    
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

    # Match the graphs where DAG is undertermined
    if graph_match:
        # Compute vertex similarity
        for v_q in query_graph.vertices:
            max_sim = 0
            for v_g in target_graph.vertices:
                sim = vertex_similarity(v_q, v_g, sigma_vertex)
                max_sim = max(max_sim, sim)  # Select the maximum similarity for the vertex
            vertex_score += max_sim

        # Compute edge similarity
        for e_q in query_graph.edges:
            max_sim = 0
            for e_g in target_graph.edges:
                sim = edge_similarity(e_q, e_g, sigma_edge)
                max_sim = max(max_sim, sim)  # Select the maximum similarity for the edge
            edge_score += max_sim
    else:
        for idx in range(len(query_graph.vertices)):
            sim = vertex_similarity(query_graph.vertices[idx], target_graph.vertices[idx], sigma_vertex)
            vertex_score += sim

        # Compute edge similarity
        for idx in range(len(query_graph.edges)):
            sim = edge_similarity(query_graph.edges[idx], target_graph.edges[idx], sigma_edge)
            edge_score += sim

    # Normalize scores by the number of vertices and edges in the query graph
    if query_graph.vertices:
        vertex_score /= len(query_graph.vertices)
    if query_graph.edges:
        edge_score /= len(query_graph.edges)

    # Return the weighted combined similarity
    return alpha * vertex_score + beta * edge_score


def vertex_similarity(v1, v2, sigma):
    """
    Compute similarity between two vertices using a Gaussian kernel.
    
    Args:
        v1: The first vertex, expected to have an 'attr' attribute.
        v2: The second vertex, expected to have an 'attr' attribute.
        sigma: The scale parameter for the Gaussian kernel.

    Returns:
        A similarity score between the two vertices.
    """
    if not hasattr(v1, 'attr') or not hasattr(v2, 'attr'):
        raise ValueError("Vertices must have 'attr' attribute for comparison.")

    # Compute the absolute difference in vertex attributes
    delta = abs(v1.attr - v2.attr)
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
    
    if best_graph is None or stage_index >= len(best_graph.stage_ratio):
        # Default fallback: assume uniform distribution
        return 1.0 / max(1, len(query_graph.vertices))
    else:
        return best_graph.stage_ratio[stage_index]


def is_tot_request(request_type: int) -> bool:
    """Check if a request type corresponds to ToT (Tree of Thoughts)."""
    # Assuming ToT requests have a specific type identifier
    # This should be coordinated with the request type definitions
    return request_type in [0, 1]  # LATENCY or THROUGHPUT


def is_deepresearch_request(request_type: int) -> bool:
    """Check if a request type corresponds to DeepResearch collective."""
    return request_type == 2  # COLLECTIVE

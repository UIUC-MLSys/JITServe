import math
import time
import threading
from typing import List
from vllm.logger import init_logger

logger = init_logger(__file__)

Defalut_ToT_Stage = 4
Defalut_ToT_Requests_Pattern = [1, 1, 3, 1]

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
        
        self._lock = threading.Lock()
        
        assert len(request_per_stage) == stage_num, "The length of request_per_stage should be equal to stage_num"
        
    def update(self) -> None:
        if len(self.current_stage_length) == self.request_per_stage[self.stage_finished]:
            self.stage_finished += 1
            stage_input_length = sum([length[0] for length in self.current_stage_length])
            stage_output_length = sum([length[1] for length in self.current_stage_length])
            current_stage_ratio = stage_output_length / stage_input_length
            self.output_input_ratio.append(current_stage_ratio)
            self.current_stage_length = []
            
            self.stage_finish_time.append(time.time() - self.current_time)
            self.current_time = time.time()
            
    def add_length(self, input_length: int, output_length: int) -> bool:
        with self._lock:
            self.current_stage_length.append((input_length, output_length))
            self.update()
        
            if self.stage_finished == self.stage_num:
                return True
            else:
                return False
        
    def convert_to_graph(self) -> Graph:
        vertices = [Vertex(1) for _ in range(self.stage_num)]
        edges = [Edge(weight) for weight in self.output_input_ratio]
        total_time = sum(self.stage_finish_time)
        assert total_time != 0, "The total_time should not be zero"
        stage_ratio = [stage_time / total_time for stage_time in self.stage_finish_time]
        logger.info(f"Graph Stage ratio: {stage_ratio}")
        
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


def predict_stage_ratio(query_graph: Graph, graph_set) -> float:
    stage = len(query_graph.vertices)
    best_graph: Graph | None = match_graph(query_graph, graph_set)
    if len(graph_set) > 0:
        logger.info(f"Graph set size: {len(graph_set)}")
        logger.info(f"Best Graph stage ratio: {best_graph.stage_ratio}")
    
    if best_graph is None:
        logger.info("Use default stage ratio")
        logger.info(f"Default stage ratio: {Defalut_ToT_Requests_Pattern[stage] / sum(Defalut_ToT_Requests_Pattern)}")
        return Defalut_ToT_Requests_Pattern[stage] / sum(Defalut_ToT_Requests_Pattern)
    else:
        logger.info(f"Best Graph stage ratio: {best_graph.stage_ratio[stage]}")
        return best_graph.stage_ratio[stage]

    
def match_graph(query_graph, graph_set, alpha=0.5, beta=0.5, sigma_vertex=1.0, sigma_edge=1.0) -> Graph:
    best_graph = None
    best_similarity = -float('inf')

    for graph in graph_set:
        similarity = compute_similarity(query_graph, graph, alpha=alpha, beta=beta, sigma_vertex=sigma_vertex, sigma_edge=sigma_edge)
        
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

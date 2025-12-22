import json
from typing import List, Optional, Tuple
from similarity import Graph, RequestApplication
from vllm.logger import init_logger

logger = init_logger("vllm")

def read_deepresearch_traces(trace_file_path: str, is_all_node: bool = False) -> List[Graph]:
    """
    Read deepresearch traces from a trace file and return a list of Graph objects.
    
    Args:
        trace_file_path: Path to the trace file (e.g., "benchmarks/dataset/trace/deepresearch_training.json")
        is_all_node: Whether to use all-node approach
        
    Returns:
        List of Graph objects constructed from the traces
    """
    try:
        with open(trace_file_path, 'r', encoding='utf-8') as f:
            deepresearch_traces = json.load(f)
        
        logger.info(f"Loaded {len(deepresearch_traces)} deepresearch traces from {trace_file_path}")
        
        graphs = []
        for trace in deepresearch_traces:
            try:
                graph = _convert_trace_to_graph(trace, is_all_node)
                graphs.append(graph)
            except Exception as e:
                logger.warning(f"Failed to convert trace {trace.get('request_id', 'unknown')} to graph: {e}")
        
        logger.info(f"Successfully converted {len(graphs)} traces to Graph objects")
        return graphs
        
    except FileNotFoundError:
        logger.error(f"Trace file not found: {trace_file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {trace_file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error reading traces from {trace_file_path}: {e}")
        return []

def _convert_trace_to_graph(trace: dict, is_all_node: bool) -> Graph:
    """
    Convert a single deepresearch trace to a Graph object.
    
    Args:
        trace: Dictionary containing the trace data
        is_all_node: Whether to use all-node approach
        
    Returns:
        Graph object with nodes and times filled from the trace
    """
    stages = trace.get("stages", [])
    
    # Fill nodes: List[List[tuple[input_len, output_len]]]
    nodes = []
    stage_max_outputs = []
    
    for stage in stages:
        stage_requests = stage.get("requests", [])
        stage_nodes = []
        stage_outputs = []
        
        if is_all_node:
            # All-node approach: include all requests as separate nodes
            for request in stage_requests:
                input_tokens = request.get("input_tokens", 0)
                output_tokens = request.get("output_tokens", 0)
                stage_nodes.append((input_tokens, output_tokens))
                stage_outputs.append(output_tokens)
        else:
            # Single-node approach: one node per stage with sum of all tokens
            total_input = sum(request.get("input_tokens", 0) for request in stage_requests)
            total_output = sum(request.get("output_tokens", 0) for request in stage_requests)
            stage_nodes.append((total_input, total_output))
            stage_outputs.append(total_output)
        
        nodes.append(stage_nodes)
        # Calculate max output length for this stage
        max_output = max(stage_outputs) if stage_outputs else 0
        stage_max_outputs.append(max_output)
    
    # Calculate times: ratio of all stages based on max output length
    times = None
    if stage_max_outputs:
        total_max_output = sum(stage_max_outputs)
        if total_max_output > 0:
            times = [max_output / total_max_output for max_output in stage_max_outputs]
        else:
            # If all max outputs are 0, use equal ratios
            num_stages = len(stage_max_outputs)
            times = [1.0 / num_stages] * num_stages
    
    return Graph(nodes=nodes, times=times, is_all_node=is_all_node, application=RequestApplication.DEEPRESEARCH)

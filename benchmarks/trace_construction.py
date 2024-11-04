import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from benchmarks.trace import Trace, TraceConfig

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def trace_construction(trace_config: TraceConfig, save_path: str):
    trace = Trace.from_config(trace_config)
    trace.save_trace(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace Construction Benchmark")
    parser.add_argument("--config_key", type=str, default='test_short', help="Configuration key")
    parser.add_argument("--save_path", type=str, default='./example.json', help="Path to save the trace")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--max_request_num", type=int, default=1000, help="Maximum number of requests to handle")
    parser.add_argument("--is_real", type=bool, default=False, help="Flag to indicate if real data is used")
    parser.add_argument("--is_random_pick", type=bool, default=False, help="Flag for random selection of requests")
    parser.add_argument("--trust_model_code", type=bool, default=False, help="Flag to trust model code or not")
    parser.add_argument("--poisson_lambda", type=float, default=80, help="Poisson lambda value")
    parser.add_argument("--deadline_range", type=int, default=(1000, 50000), help="Deadline range")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
        if args.config_key not in config:
            print("Using arguments from command line to construct the trace.")
            
            trace_config = TraceConfig(args.model, args.tokenizer, args.dataset_name, args.dataset_path, 
                                       args.max_request_num, args.is_real, args.is_random_pick, 
                                       args.trust_model_code, args.poisson_lambda, args.deadline_range, args.seed)
        else:
            print(f"Using configuration from the config file for key: {args.config_key}")
            trace_config = TraceConfig.from_dict(config[args.config_key])
    
    print(f"Trace Configuration: {trace_config}")        
    trace_construction(trace_config, args.save_path)
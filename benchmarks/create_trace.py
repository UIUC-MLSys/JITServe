#!/usr/bin/env python3

import json
import argparse
import os

def parse_request_ratio(ratio_str):
    """Parse request ratio string in format 'num_1,num_2,num_3'"""
    try:
        ratios = [float(x.strip()) for x in ratio_str.split(',')]
        if len(ratios) != 3:
            raise ValueError("Request ratio must have exactly 3 numbers")
        return ratios
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid request ratio format: {e}")

def calculate_request_counts(num_prompts, ratios):
    """Calculate number of requests for each type based on ratios"""
    latency_ratio, throughput_ratio, collective_ratio = ratios
    
    # Normalize ratios: collective requests are weighted by 10 (each collective = 10 single requests)
    # So the actual ratio for request counts is latency:throughput:collective/10
    effective_collective_ratio = collective_ratio / 7
    total_ratio = latency_ratio + throughput_ratio + effective_collective_ratio
    
    # Calculate counts
    latency_count = int(num_prompts * (latency_ratio / total_ratio))
    throughput_count = int(num_prompts * (throughput_ratio / total_ratio))
    collective_count = int(num_prompts * (effective_collective_ratio / total_ratio))
    
    # Adjust for any rounding differences
    actual_total = latency_count + throughput_count + collective_count
    if actual_total < num_prompts:
        # Add remaining to latency type
        latency_count += num_prompts - actual_total
    
    return latency_count, throughput_count, collective_count

def load_lmsys_trace(trace_path):
    """Load the lmsys.json trace file"""
    with open(trace_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_modified_trace(original_trace, latency_count, throughput_count, collective_count):
    """Create new trace with modified request types"""
    modified_trace = []
    
    # We need half the collective requests to be ToT (only include ToT in traces)
    tot_count = collective_count // 2
    
    total_needed = latency_count + throughput_count + tot_count
    
    # Use original trace as base, cycling through if needed
    trace_index = 0
    
    # Add latency requests (request_type = 0)
    for i in range(latency_count):
        request = original_trace[trace_index % len(original_trace)].copy()
        request['request_type'] = 0  # latency type
        modified_trace.append(request)
        trace_index += 1
    
    # Add throughput requests (request_type = 1)  
    for i in range(throughput_count):
        request = original_trace[trace_index % len(original_trace)].copy()
        request['request_type'] = 1  # throughput type
        modified_trace.append(request)
        trace_index += 1
    
    # Add ToT requests (request_type = 2)
    for i in range(tot_count):
        request = original_trace[trace_index % len(original_trace)].copy()
        request['request_type'] = 2  # ToT type
        modified_trace.append(request)
        trace_index += 1
    
    return modified_trace

def save_trace(trace, output_path):
    """Save the modified trace to output file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Create modified trace based on request ratios')
    parser.add_argument('--request_ratio', type=parse_request_ratio, required=True,
                       help='Request ratios in format "num_1,num_2,num_3" for latency:throughput:collective')
    parser.add_argument('--num_prompts', type=int, required=True,
                       help='Total number of prompts to generate')
    
    args = parser.parse_args()
    
    # Calculate request counts
    latency_count, throughput_count, collective_count = calculate_request_counts(
        args.num_prompts, args.request_ratio
    )
    
    # Calculate deepresearch requests (the other half of collective)
    deepresearch_count = collective_count - (collective_count // 2)
    
    print(f"Request counts - Latency: {latency_count}, Throughput: {throughput_count}, Collective: {collective_count}")
    print(f"ToT requests (half of collective): {collective_count // 2}")
    print(f"Deepresearch requests: {deepresearch_count}")
    
    # Load original trace
    trace_path = os.path.join(os.path.dirname(__file__), 'dataset', 'trace', 'lmsys.json')
    original_trace = load_lmsys_trace(trace_path)
    
    # Create modified trace
    modified_trace = create_modified_trace(original_trace, latency_count, throughput_count, collective_count)
    
    # Generate output filename
    ratio_str = '_'.join(str(r) for r in args.request_ratio)
    output_filename = f"lmsys_num_prompts_{args.num_prompts}_request_ratio_{ratio_str}.json"
    output_path = os.path.join(os.path.dirname(__file__), 'dataset', 'trace', output_filename)
    
    # Check if output file already exists
    if os.path.exists(output_path):
        print(f"Output file already exists, skipping update.")
        print(f"Trace file: {output_path}")
        print(f"Number of deepresearch requests calculated: {deepresearch_count}")
        return
    
    # Save modified trace
    save_trace(modified_trace, output_path)
    
    print(f"Created trace with {len(modified_trace)} requests")
    print(f"Trace file: {output_path}")
    print(f"Number of deepresearch requests calculated: {deepresearch_count}")

if __name__ == "__main__":
    main()
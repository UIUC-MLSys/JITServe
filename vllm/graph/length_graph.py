import os
import json
import time
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from similarity import Graph, Edge, Vertex, match_graph
# TODO: add plot request stage ratios function
from plot_request_stage_ratios import plot_stage_ratios

def compute_average_normalized_list(normalized_lists):
    # Ensure input is not empty
    if len(normalized_lists) == 0:
        raise ValueError("The input list is empty.")

    # Check if all lists are normalized
    for lst in normalized_lists:
        if not np.all(sum(lst)):
            raise ValueError("All input lists must be normalized (sum to 1).")

    # Convert to numpy array for element-wise operations
    array = np.array(normalized_lists)

    # Calculate element-wise average
    average = np.mean(array, axis=0)

    # Normalize the resulting list to ensure it sums to 1
    normalized_average = average / np.sum(average)

    return normalized_average.tolist()

def construct_graph_set_super(input_data, output_data, times, ratios):
    assert len(input_data) == len(times)
    assert len(output_data) == len(ratios)
    elapsed_times = []
    graph_set = set()
    for inputs, outputs, time1, ratio in zip(input_data, output_data, times, ratios):
        start_time = time.time()
        nodes = []
        edges = []
        # Construct the first stage of the graph with a single node
        nodes.append(Vertex(1))
        edges.append(Edge((sum(outputs[:3])/inputs[0])))
        for idx in range(1, len(inputs), 2):
            nodes.append(Vertex(1))
            edges.append(Edge((sum(outputs[idx*3:idx*3+6])/sum(inputs[idx:idx+2]))))
        graph_set.add(Graph(nodes=nodes, edges=edges, total_time=time1, stage_ratio=ratio))
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        elapsed_times.append(elapsed_time)
    
    # Print the average time taken to construct a graph
    print(f"Average time taken to construct a super-node graph: {np.mean(elapsed_times)} seconds")
    return graph_set

def construct_graph_set_all_node(input_data, output_data, times, ratios):
    assert len(input_data) == len(times)
    assert len(output_data) == len(ratios)
    elapsed_times = []
    graph_set = set()
    for inputs, outputs, time1, ratio in zip(input_data, output_data, times, ratios):
        start_time = time.time()
        nodes = []
        edges = []
        for idx in range(len(inputs)):
            nodes.append(Vertex(1))
            edges.append(Edge(outputs[idx*3]/inputs[idx]))
            nodes.append(Vertex(1))
            edges.append(Edge(outputs[idx*3+1]/inputs[idx]))
            nodes.append(Vertex(1))
            edges.append(Edge(outputs[idx*3+2]/inputs[idx]))
        graph_set.add(Graph(nodes=nodes, edges=edges, total_time=time1, stage_ratio=ratio))
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        elapsed_times.append(elapsed_time)

    # Print the average time taken to construct a graph
    print(f"Average time taken to construct an all-node graph: {np.mean(elapsed_times)} seconds")
    return graph_set

def match_graph_set(stages_train_set: set[Graph], stages_test_set: set[Graph], stage_dir: str):
    entry_match_results = []
    time_benchmark = {}
    for i, entry in enumerate(stages_test_set):
        matched_graphs = []
        for stage_idx in range(len(entry.nodes)-1):
            start_time = time.time()

            new_graph = Graph(
                nodes=entry.nodes[:stage_idx + 1],
                edges=entry.edges[:stage_idx + 1],
                total_time=entry.total_time,
                stage_ratio=entry.stage_ratio,
            )
            matched_graphs.append(match_graph(new_graph, stages_train_set))

            elapsed_time = time.time() - start_time  # Calculate elapsed time
            if f"Request_{i + 1}" not in time_benchmark:
                time_benchmark[f"Request_{i + 1}"] = []
            time_benchmark[f"Request_{i + 1}"].append(elapsed_time)
        entry_match_result = {
            "original_graph": entry,
            "matched_graphs": matched_graphs,
        }
        entry_match_results.append(entry_match_result)
    
    # Save the benchmark times to a JSON file
    time_benchmark_path = os.path.join(stage_dir, "super_node_time_benchmark.json")
    with open(time_benchmark_path, "w") as json_file:
        json.dump(time_benchmark, json_file, indent=4)

    all_time_lists = [entry for entry in time_benchmark.values()]

    # Transpose the list of lists using zip to group by index
    restructured_lists = list(zip(*all_time_lists))

    # Print the mean of each restructured list
    for idx, time_group in enumerate(restructured_lists):
        print(f"SuperNode: Mean of time stage {idx + 1}: {np.mean(time_group):.4f} seconds")

    return entry_match_results

def match_graph_set_all_node(stages_train_set: set[Graph], stages_test_set: set[Graph], stage_dir: str):
    entry_match_results = []
    time_benchmark = {}
    for i, entry in enumerate(stages_test_set):
        matched_graphs = []
        for stage_idx in range(0, len(entry.nodes)-3, 3):
            start_time = time.time()

            new_graph = Graph(
                nodes=entry.nodes[:stage_idx + 1],
                edges=entry.edges[:stage_idx + 1],
                total_time=entry.total_time,
                stage_ratio=entry.stage_ratio,
            )
            matched_graphs.append(match_graph(new_graph, stages_train_set))

            elapsed_time = time.time() - start_time  # Calculate elapsed time
            if f"Request_{i + 1}" not in time_benchmark:
                time_benchmark[f"Request_{i + 1}"] = []
            time_benchmark[f"Request_{i + 1}"].append(elapsed_time)
        entry_match_result = {
            "original_graph": entry,
            "matched_graphs": matched_graphs,
        }
        entry_match_results.append(entry_match_result)

    # Save the benchmark times to a JSON file
    time_benchmark_path = os.path.join(stage_dir, "all_node_time_benchmark.json")
    with open(time_benchmark_path, "w") as json_file:
        json.dump(time_benchmark, json_file, indent=4)

    all_time_lists = [entry for entry in time_benchmark.values()]

    # Transpose the list of lists using zip to group by index
    restructured_lists = list(zip(*all_time_lists))

    # Print the mean of each restructured list
    for idx, time_group in enumerate(restructured_lists):
        print(f"All Node: Mean of time stage {idx + 1}: {np.mean(time_group):.4f} seconds")

    return entry_match_results

def plot_graph_match_results(entry_match_results, average_stage_ratio, stage_dir_name, figures_dir_name, print_results=False):
    original_graph_ratios = [entry["original_graph"].stage_ratio for entry in entry_match_results]
    matched_graph_ratios = [entry["matched_graphs"] for entry in entry_match_results]
    for i, entry_match_result in enumerate(entry_match_results):
        original_graph = entry_match_result["original_graph"]
        matched_graphs = entry_match_result["matched_graphs"]
        if print_results:
            print("Original graph:")
            print(original_graph)
            print("Matched graphs:")
            for matched_graph in matched_graphs:
                print(matched_graph)
        
        # Plot the original graph and matched graphs
        plot_stage_ratios(
            average_stage_ratio=average_stage_ratio,
            real_stage_ratio=original_graph.stage_ratio,
            matched_stage_ratios=[matched_graph.stage_ratio for matched_graph in matched_graphs],
            title=f"Request_{i + 1}",
            stage_dir_name=stage_dir_name,
            figures_dir_name=figures_dir_name,
        )
    
    # Plot the average stage ratio
    plot_stage_ratios(
        average_stage_ratio=average_stage_ratio,
        real_stage_ratio=compute_average_normalized_list(original_graph_ratios),
        matched_stage_ratios = [
            compute_average_normalized_list(stage_ratios)
            for stage_ratios in zip(*[[matched_graph.stage_ratio for matched_graph in matched_graphs] for matched_graphs in matched_graph_ratios])
        ],
        title="Average_test_request",
        stage_dir_name=stage_dir_name,
        figures_dir_name=figures_dir_name,
    )

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and match graph datasets.")
    parser.add_argument(
        "--print_results",
        action="store_true",
        help="Whether to print graph match results in the console.",
    )
    args = parser.parse_args()
    print_results = args.print_results

    # Base directory
    base_dir = "."  # Replace with the base directory path if needed

    # Find all files starting with "len-" and ending with ".json"
    stage_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.endswith("-stage") and os.path.isdir(os.path.join(base_dir, d))]

    # Process each stage project
    for stage_dir in stage_dirs:
        json_files = [f for f in os.listdir(stage_dir) if f.endswith(".json") and f.startswith("len_")]
        lengths = []
        profiles = []
        for json_file in json_files:
            json_file_path = os.path.join(stage_dir, json_file)
            try:
                # Read the JSON file
                with open(json_file_path, "r") as file:
                    data = json.load(file)
                if "length" in data:
                    lengths.append(data["length"])
                if "profile" in data:
                    profiles.append(data["profile"])
                else:
                    print(f"File {json_file} does not contain 'length' or 'profile'. Skipping.")
                    continue
            except Exception as e:
                print(f"An error occurred while processing {json_file}: {e}")
                continue

        # If no lengths were found, skip creating the dataset
        if not lengths or not profiles:
            print(f"No valid lengths/profiles found in {json_file}. Skipping dataset creation.")
            continue

        # Extract stage_time and round_time for regression and prediction
        input_lengths = np.array([
            [length for entry in arr for length in entry["input_length"]] for arr in lengths
        ])

        output_lengths = np.array([
            [length for entry in arr for length in entry["output_length"]] for arr in lengths
        ])

        print(f"Input lengths shape: {input_lengths.shape}, Output lengths shape: {output_lengths.shape}")

        total_times = np.array([profile["total_time"] for profile in profiles])
        time_per_stage_ratios = np.array([profile["time_per_stage_ratio"] for profile in profiles])

        print(f"Stage times shape: {input_lengths.shape}")

        # Create a single set of train-test split indices
        train_indices, test_indices = train_test_split(
            np.arange(len(input_lengths)), test_size=0.2, random_state=42
        )

        # Use the indices to split the data
        input_train, input_test = input_lengths[train_indices], input_lengths[test_indices]
        output_train, output_test = output_lengths[train_indices], output_lengths[test_indices]
        total_train, total_test = total_times[train_indices], total_times[test_indices]
        ratio_train, ratio_test = time_per_stage_ratios[train_indices], time_per_stage_ratios[test_indices]
        if (print_results):
            print(f"Stage train shape: {input_train.shape}, Stage test shape: {input_test.shape}")
            print(f"Round train shape: {output_train.shape}, Round test shape: {output_test.shape}")
            print(f"Total train shape: {total_train.shape}, Total test shape: {total_test.shape}")
            print(f"Ratio train shape: {ratio_train.shape}, Ratio test shape: {ratio_test.shape}")

        # Train models and predict for stage_times
        avg_ratio_train = compute_average_normalized_list(ratio_train)
        stages_train_set = construct_graph_set_super(input_train, output_train, total_train, ratio_train)
        stages_test_set = construct_graph_set_super(input_test, output_test, total_test, ratio_test)
        entry_match_results = match_graph_set(stages_train_set, stages_test_set, stage_dir)
        plot_graph_match_results(entry_match_results, avg_ratio_train, stage_dir, "length_ratio_super", print_results)

        # Train models and predict for round_times
        avg_ratio_train = compute_average_normalized_list(ratio_train)
        rounds_train_set = construct_graph_set_all_node(input_train, output_train, total_train, ratio_train)
        rounds_test_set = construct_graph_set_all_node(input_test, output_test, total_test, ratio_test)
        entry_match_results = match_graph_set_all_node(rounds_train_set, rounds_test_set, stage_dir)
        plot_graph_match_results(entry_match_results, avg_ratio_train, stage_dir, "length_ratio_all", print_results)

if __name__ == "__main__":
    main()
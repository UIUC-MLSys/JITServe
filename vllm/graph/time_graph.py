import os
import json
import time
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
# TODO: add compare similarity function and plot request stage ratios function
from compare_similarity import Graph, Edge, Vertex, match_graph
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

def construct_graph_set(datas, times, ratios):
    assert len(datas) == len(times)
    assert len(datas) == len(ratios)
    graph_set = set()
    for entry, time, ratio in zip(datas, times, ratios):
        nodes = []
        edges = []
        for node in entry:
            nodes.append(Vertex(1))
            edges.append(Edge(node))
        graph_set.add(Graph(nodes=nodes, edges=edges, total_time=time, stage_ratio=ratio))
    return graph_set

def match_graph_set(stages_train_set: set[Graph], stages_test_set: set[Graph]):
    entry_match_results = []
    for entry in stages_test_set:
        matched_graphs = []
        for stage_idx in range(len(entry.nodes)-1):
            new_graph = Graph(
                nodes=entry.nodes[:stage_idx + 1],
                edges=entry.edges[:stage_idx + 1],
                total_time=entry.total_time,
                stage_ratio=entry.stage_ratio,
            )
            matched_graphs.append(match_graph(new_graph, stages_train_set))
        entry_match_result = {
            "original_graph": entry,
            "matched_graphs": matched_graphs,
        }
        entry_match_results.append(entry_match_result)
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
        profiles = []
        for json_file in json_files:
            json_file_path = os.path.join(stage_dir, json_file)
            try:
                # Read the JSON file
                with open(json_file_path, "r") as file:
                    data = json.load(file)
                if "profile" in data:
                    profiles.append(data["profile"])
                else:
                    print(f"File {json_file} does not contain 'profile'. Skipping.")
                    continue
            except Exception as e:
                print(f"An error occurred while processing {json_file}: {e}")
                continue

        # If no profiles were found, skip creating the dataset
        if not profiles:
            print(f"No valid profiles found in {json_file}. Skipping dataset creation.")
            continue

        # Extract stage_time and round_time for regression and prediction
        stage_times = np.array([
            [
                *profile["generation_time"],
                *profile["evaluation_time"]
            ] for profile in profiles
        ])
        round_times = np.array([profile["round_time"] for profile in profiles])
        total_times = np.array([profile["total_time"] for profile in profiles])
        time_per_stage_ratios = np.array([profile["time_per_stage_ratio"] for profile in profiles])

        print(f"Stage times shape: {stage_times.shape}")

        # Parameterize stage_times
        stage_times = np.array([
            np.array(entry[1:]) / entry[0] if entry[0] != 0 else entry[1:]
            for entry in stage_times
        ])

        # Parameterize round_times
        round_times = np.array([
            entry[1:] / entry[0] if entry[0] != 0 else entry[1:]
            for entry in round_times
        ])

        # Create a single set of train-test split indices
        train_indices, test_indices = train_test_split(
            np.arange(len(stage_times)), test_size=0.2, random_state=42
        )

        # Use the indices to split the data
        stage_train, stage_test = stage_times[train_indices], stage_times[test_indices]
        round_train, round_test = round_times[train_indices], round_times[test_indices]
        total_train, total_test = total_times[train_indices], total_times[test_indices]
        ratio_train, ratio_test = time_per_stage_ratios[train_indices], time_per_stage_ratios[test_indices]
        if (print_results):
            print(f"Stage train shape: {stage_train.shape}, Stage test shape: {stage_test.shape}")
            print(f"Round train shape: {round_train.shape}, Round test shape: {round_test.shape}")
            print(f"Total train shape: {total_train.shape}, Total test shape: {total_test.shape}")
            print(f"Ratio train shape: {ratio_train.shape}, Ratio test shape: {ratio_test.shape}")

        # Train models and predict for stage_times
        avg_ratio_train = compute_average_normalized_list(ratio_train)
        stages_train_set = construct_graph_set(stage_train, total_train, ratio_train)
        stages_test_set = construct_graph_set(stage_test, total_test, ratio_test)
        entry_match_results = match_graph_set(stages_train_set, stages_test_set)
        plot_graph_match_results(entry_match_results, avg_ratio_train, stage_dir, "stage_times_ratio", print_results)

        # Train models and predict for round_times
        avg_ratio_train = compute_average_normalized_list(ratio_train)
        rounds_train_set = construct_graph_set(round_train, total_train, ratio_train)
        rounds_test_set = construct_graph_set(round_test, total_test, ratio_test)
        entry_match_results = match_graph_set(rounds_train_set, rounds_test_set)
        plot_graph_match_results(entry_match_results, avg_ratio_train, stage_dir, "round_times_ratio", print_results)

if __name__ == "__main__":
    main()
#!/bin/bash
policies=("jitserve" "ltr" "fcfs" "vllm" "srtf" "autellix")
rates=(6.0)
batch_sizes=(32)
penalty_factors=(100)
output_dir="batch_result/test"
model="meta-llama/Llama-3.1-8B-Instruct"

# trace config paths
default_trace_file="traces/lmsys.json"
deepresearch_trace_file="traces/deepresearch_filter.jsonl"
request_ratios="1,1,1"
num_prompts="200"
use_all_node="false"

mkdir -p "$output_dir"

# cleanup function to handle interrupts
cleanup() {
    echo -e "\nInterrupted. Cleaning up..."
    kill $server_pid 2>/dev/null
    kill $scheduler_pid 2>/dev/null
    kill $deepresearch_pid 2>/dev/null
    exit 1
}

trap cleanup SIGINT

# Generate trace file if request_ratios and num_prompts are provided
trace_file="$default_trace_file"
num_deep_research=0
rate_deepresearch=0
rate_scheduler=0

if [[ -n "$request_ratios" && -n "$num_prompts" ]]; then
    echo "Using custom request ratios. Generating trace file..."
    
    create_output=$(python3 benchmark/trace/tools/create_trace.py --request_ratio "$request_ratios" --num_prompts "$num_prompts" 2>&1)
    echo "$create_output"

    trace_file=$(echo "$create_output" | grep "Trace file:" | sed 's/Trace file: //')
    num_deep_research=$(echo "$create_output" | grep "Number of deepresearch requests calculated:" | sed 's/Number of deepresearch requests calculated: //')
    
    if [[ -z "$trace_file" || -z "$num_deep_research" ]]; then
        echo "Error: failed to parse create_trace.py output"
        echo "Output: $create_output"
        exit 1
    fi
    
    echo "Using trace file: $trace_file"
    echo "Deepresearch request count: $num_deep_research"
else
    echo "Using default trace file: $trace_file"
    if [[ -n "$num_prompts" ]]; then
        echo "Warning: num_prompts set without request_ratios, using default trace"
    fi
fi

start_server() {
    local policy_name="$1"
    local policy_arg="$policy_name"
    local chunked_prefill="True"
    local prediction_arg=""

    case "$policy_name" in
        vllm)
            policy_arg="fcfs"
            chunked_prefill="False"
            ;;
        oracle)
            policy_arg="jitserve"
            prediction_arg="--disable-prediction"
            ;;
    esac

    python3 -m jitserve.server \
        --scheduling-policy "$policy_arg" \
        $prediction_arg \
        --enable-chunked-prefill "$chunked_prefill" \
        --penalty-factor "$penalty_factor" \
        --max-num-seqs "$batch_size" \
        $extra_args \
        --model "$model" &
}

run_tot_benchmark() {
    local policy_name="$1"
    local arrival_rate="$2"
    local output_file="$3"
    local prompts_arg=("--num-prompts" "$num_lmsys")

    if [[ -z "$num_prompts" ]]; then
        prompts_arg=("--num-prompts" "200")
    fi

    python3 benchmark/benchmark_scheduler.py \
        --model "$model" \
        --policy "$policy_name" \
        --arrival-rate "$arrival_rate" \
        --penalty-factor "$penalty_factor" \
        --batch-size "$batch_size" \
        --slo-constraint "0.8,0.08,8" \
        "${prompts_arg[@]}" \
        --trace-path "$trace_file" > "$output_file" 2>&1 &
}

run_deepresearch_benchmark() {
    local policy_name="$1"
    local output_file="$2"

    python3 benchmark/benchmark_scheduler_deepresearch.py \
        --model "$model" \
        --policy "$policy_name" \
        --arrival-rate "$rate_deepresearch" \
        --penalty-factor "$penalty_factor" \
        --slo-constraint "0.8,0.08,8" \
        --num-prompts "$num_deep_research" \
        --trace-path "$deepresearch_trace_file" > "$output_file" 2>&1 &
}

for rate in "${rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for penalty_factor in "${penalty_factors[@]}"; do
            for policy in "${policies[@]}"; do
                echo -e "\n========================================"
                echo "Starting test: policy=$policy, rate=$rate, batch_size=$batch_size, penalty_factor=$penalty_factor"
                echo "========================================"

                if [[ $num_deep_research -gt 0 ]]; then
                    # compute deepresearch arrival rate
                    rate_deepresearch=$(python3 -c "print($rate * ($num_deep_research / $num_prompts))")
                    rate_scheduler=$(python3 -c "print($rate - $rate_deepresearch)")
                    # compute lmsys prompt count
                    num_lmsys=$(python3 -c "print($num_prompts - $num_deep_research)")
                    echo "Deepresearch arrival rate: $rate_deepresearch"
                    echo "Scheduler arrival rate: $rate_scheduler"
                    echo "LMSYS prompt count: $num_lmsys"
                else
                    rate_scheduler=$rate
                    num_lmsys=$num_prompts
                    echo "Scheduler arrival rate: $rate_scheduler"
                    echo "LMSYS prompt count: $num_lmsys"
                fi

                extra_args=""
                if [ "$use_all_node" = "true" ]; then
                    extra_args="$extra_args --use-all-node"
                fi

                echo "Starting server..."
                start_server "$policy"
                server_pid=$!
                echo "Waiting for server init (40s)..."
                sleep 40
                echo "Starting client..."
                
                # benchmark_scheduler.py
                scheduler_output_file="${output_dir}/scheduler_${policy}_${rate}_${batch_size}_${penalty_factor}.log"
                if [[ -n "$request_ratios" ]]; then
                    run_tot_benchmark "$policy" "$rate_scheduler" "$scheduler_output_file"
                else
                    # use default trace
                    run_tot_benchmark "$policy" "$rate" "$scheduler_output_file"
                fi
                scheduler_pid=$!

                # benchmark_scheduler_deepresearch.py
                if [[ $num_deep_research -gt 0 ]]; then
                    deepresearch_output_file="${output_dir}/deepresearch_${policy}_${rate}_${batch_size}_${penalty_factor}.log"
                    run_deepresearch_benchmark "$policy" "$deepresearch_output_file"
                    deepresearch_pid=$!
                    
                    echo "Waiting for both benchmarks to complete..."
                    wait $scheduler_pid
                    wait $deepresearch_pid
                else
                    echo "Waiting for scheduler benchmark to complete..."
                    wait $scheduler_pid
                fi

                echo "Stopping server..."
                kill $server_pid
                wait $server_pid 2>/dev/null
                echo "Cooling down (30s)..."
                sleep 30
            done
        done
    done
done

echo -e "\nAll tests completed! Results saved in $output_dir/"

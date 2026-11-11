#!/bin/bash
# Multi-replica goodput (paper Fig 18, §4.3).
# Each replica is an independent jitserve.server (its own vLLM engine and
# scheduler) on `tensor_parallel` GPUs; jitserve.router in front places every
# request on one replica. Data parallelism = number of replicas.
# Each run is "policy|routing": JITServe uses power-of-K, baselines round-robin;
# "jitserve|round-robin" isolates the effect of power-of-K placement.
# Arrival rate and prompt count scale with the number of replicas.
runs=("jitserve|power-of-k" "jitserve|round-robin" "ltr|round-robin" "fcfs|round-robin")
replica_counts=(2)
tensor_parallel=2          # replica i runs on GPUs i*tp .. i*tp+tp-1
rate_per_replica=6.0
duration_s=1200            # arrival window; num_prompts = rate * duration
power_of_k=2
batch_size=32
max_num_seqs=128           # same running cap for every policy
gmax_window=32             # GMAX window B (JITServe only)
penalty_factor=100

model="Qwen/Qwen2.5-14B-Instruct"
qrf_model="assets/qrf/qrf_model/0_qrf_lmsys_chat_qwen25_7b.pkl"
qrf_vectorizer="assets/qrf/qrf_vectorizer/0_qrf_lmsys_chat_qwen25_7b.pkl"
output_dir="batch_result/e2e-multi-replica/${model##*/}/"

deepresearch_trace_file="traces/deepresearch_filter.jsonl"
request_ratios="1,1,1"
base_port=8001
base_vllm_port=29500       # torch.distributed ports, one range per replica

mkdir -p "$output_dir/server_logs"

pids=()
cleanup() {
    echo -e "\nCleaning up..."
    kill "${pids[@]}" $router_pid $scheduler_pid $deepresearch_pid $prediction_pid 2>/dev/null
    exit 1
}
trap cleanup SIGINT SIGTERM

wait_healthy() {
    local url="$1" pid="$2"
    until curl -sf "$url/health" >/dev/null; do
        kill -0 "$pid" 2>/dev/null || { echo "Process for $url exited during init"; cleanup; }
        sleep 5
    done
}

num_gpus=$(nvidia-smi -L | wc -l)

# One shared QRF prediction server for the router and all replicas.
python3 -m jitserve.request_analyzer.prediction \
    --model-path "$qrf_model" --tokenizer-path "$qrf_vectorizer" &
prediction_pid=$!
until (echo > /dev/tcp/127.0.0.1/65433) 2>/dev/null; do
    kill -0 $prediction_pid 2>/dev/null || { echo "Prediction server exited"; exit 1; }
    sleep 3
done

for num_replicas in "${replica_counts[@]}"; do
    if (( num_replicas * tensor_parallel > num_gpus )); then
        echo "Skipping $num_replicas replicas x TP $tensor_parallel: only $num_gpus GPUs"
        continue
    fi
    rate=$(python3 -c "print($rate_per_replica * $num_replicas)")
    num_prompts=$(python3 -c "print(int($rate * $duration_s))")
    create_output=$(python3 benchmark/trace/tools/create_trace.py --request_ratio "$request_ratios" --num_prompts "$num_prompts" 2>&1)
    trace_file=$(echo "$create_output" | grep "Trace file:" | sed 's/Trace file: //')
    num_deep_research=$(echo "$create_output" | grep "Number of deepresearch requests calculated:" | sed 's/Number of deepresearch requests calculated: //')
    if [[ -z "$trace_file" || -z "$num_deep_research" ]]; then
        echo "Error: failed to parse create_trace.py output"; echo "$create_output"; cleanup
    fi
    rate_deepresearch=$(python3 -c "print($rate * ($num_deep_research / $num_prompts))")
    rate_scheduler=$(python3 -c "print($rate - $rate_deepresearch)")
    num_lmsys=$((num_prompts - num_deep_research))

    for run in "${runs[@]}"; do
        IFS='|' read -r policy routing <<< "$run"
        echo -e "\n========================================"
        echo "policy=$policy routing=$routing replicas=$num_replicas tp=$tensor_parallel rate=$rate prompts=$num_prompts"
        echo "========================================"
        chunked_prefill="True"
        policy_arg="$policy"
        [ "$policy" = "vllm" ] && { policy_arg="fcfs"; chunked_prefill="False"; }
        tag="${policy}_${routing}_${rate}_${num_replicas}x${tensor_parallel}"

        pids=()
        replica_urls=""
        for ((i = 0; i < num_replicas; i++)); do
            port=$((base_port + i))
            gpus=$(seq -s, $((i * tensor_parallel)) $((i * tensor_parallel + tensor_parallel - 1)))
            CUDA_VISIBLE_DEVICES=$gpus VLLM_PORT=$((base_vllm_port + i * 50)) \
            python3 -m jitserve.server \
                --port "$port" \
                --scheduling-policy "$policy_arg" \
                --enable-chunked-prefill "$chunked_prefill" \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$max_num_seqs" \
                --gmax-window "$gmax_window" \
                --tensor-parallel-size "$tensor_parallel" \
                --disable-custom-all-reduce \
                --prediction-model-path "$qrf_model" \
                --prediction-tokenizer-path "$qrf_vectorizer" \
                --model "$model" > "${output_dir}/server_logs/${tag}_r${i}.log" 2>&1 &
            pids+=($!)
            replica_urls="${replica_urls:+$replica_urls,}http://localhost:$port"
        done
        for ((i = 0; i < num_replicas; i++)); do
            wait_healthy "http://localhost:$((base_port + i))" "${pids[$i]}"
        done

        python3 -m jitserve.router --port 8000 --replicas "$replica_urls" \
            --routing "$routing" --k "$power_of_k" --model "$model" \
            > "${output_dir}/server_logs/${tag}_router.log" 2>&1 &
        router_pid=$!
        wait_healthy "http://localhost:8000" "$router_pid"

        python3 benchmark/benchmark_scheduler.py \
            --model "$model" --policy "$policy" \
            --arrival-rate "$rate_scheduler" \
            --penalty-factor "$penalty_factor" --batch-size "$batch_size" \
            --slo-constraint "0.8,0.08,8" \
            --num-prompts "$num_lmsys" --trace-path "$trace_file" \
            > "${output_dir}/scheduler_${tag}.log" 2>&1 &
        scheduler_pid=$!
        python3 benchmark/benchmark_scheduler_deepresearch.py \
            --model "$model" --policy "$policy" \
            --arrival-rate "$rate_deepresearch" \
            --penalty-factor "$penalty_factor" \
            --slo-constraint "0.8,0.08,8" \
            --num-prompts "$num_deep_research" \
            --trace-path "$deepresearch_trace_file" \
            > "${output_dir}/deepresearch_${tag}.log" 2>&1 &
        deepresearch_pid=$!
        wait $scheduler_pid
        wait $deepresearch_pid

        kill $router_pid "${pids[@]}"
        wait $router_pid "${pids[@]}" 2>/dev/null
        # TP workers can outlive their server; the next run needs the GPUs empty.
        for _ in $(seq 24); do
            [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
            sleep 5
        done
        nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null
        sleep 30
    done
done

kill $prediction_pid
echo -e "\nAll tests completed! Results saved in $output_dir/"

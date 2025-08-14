#!/bin/bash

export HF_TOKEN=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

# 定义测试参数组合
test_cases=(
    # arrival_rate penalty_factor slo_constraint num_prompts
    "0.3 1 2,0.1,20 200"
)

policies=(
    "fcfs"
    "concord-default-structure"
    "concord-total-deadline-no-graph"
    "concord-static-default-structure"
    "concord-static-total-deadline"
    "concord-online-graph"
    "concord-precise"
)
output_dir="benchmark_results"
model="meta-llama/Llama-3.1-8B-Instruct"
trace_path="benchmarks/dataset/deepresearch_trace_filtered_8192.jsonl"

# 创建输出目录
mkdir -p "$output_dir"

# 定义清理函数
cleanup() {
    echo -e "\n捕获中断，正在清理..."
    kill $server_pid 2>/dev/null
    exit 1
}

# 注册中断处理
trap cleanup SIGINT

for test_case in "${test_cases[@]}"; do
    # 解析参数组合
    read -r arrival_rate penalty_factor slo_constraint num_prompts <<< "$test_case"
    
    for policy in "${policies[@]}"; do
        echo -e "\n========================================"
        echo "开始测试：策略=$policy, 速率=$arrival_rate"
        echo "惩罚因子=$penalty_factor, SLO约束=$slo_constraint, 提示数=$num_prompts"
        echo "========================================"

        # 启动服务器
        echo "启动服务器..."
        if [ "$policy" = "fcfs" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "fcfs" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs 32 \
                --model "$model" &
        elif [ "$policy" = "concord-default-structure" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs 32 \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "none" \
                --model "$model" &
        elif [ "$policy" = "concord-total-deadline-no-graph" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs 32 \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "none" \
                --use-total-deadline \
                --model "$model" &
        elif [ "$policy" = "concord-static-default-structure" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs 32 \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "static" \
                --model "$model" &
        elif [ "$policy" = "concord-static-total-deadline" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs 32 \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "static" \
                --use-total-deadline \
                --model "$model" &
        elif [ "$policy" = "concord-online-graph" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs 32 \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "online" \
                --model "$model" &
        elif [ "$policy" = "concord-precise" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --disable-prediction \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs 32 \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "precise" \
                --model "$model" &
        else
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "$policy" \
                --enable-chunked-prefill True \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs 32 \
                --model "$model" &
        fi
        server_pid=$!

        # 等待服务器启动
        echo "等待服务器初始化（60秒）..."
        sleep 45

        # 运行性能测试
        echo "启动客户端..."
        output_file="${output_dir}/deepresearch_rate${arrival_rate}_${policy}.log"
        python3 benchmarks/benchmark_scheduler_deepresearch.py \
            --model "$model" \
            --policy "$policy" \
            --arrival-rate "$arrival_rate" \
            --trace-path "$trace_path" \
            --penalty-factor "$penalty_factor" \
            --slo-constraint "$slo_constraint" \
            --burst True \
            --num-prompts "$num_prompts" \
            --max-output-len 1024 \
            --metric-percentiles "1,25,50,90,95,99" > "$output_file" 2>&1

        # 停止服务器
        echo "停止服务器..."
        kill $server_pid
        wait $server_pid 2>/dev/null

        # 清理间隔
        echo "等待系统冷却（30秒）..."
        sleep 30
    done
done

echo -e "\n所有测试完成！结果保存在 $output_dir/ 目录"
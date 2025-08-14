#!/bin/bash

export HF_TOKEN=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

# 定义测试参数组合
test_cases=(
    # batch_size arrival_rate penalty_factor slo_constraint num_prompts
    "16 0.7 1 2,0.1,20 200"
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
    read -r batch_size arrival_rate penalty_factor slo_constraint num_prompts <<< "$test_case"
    
    for policy in "${policies[@]}"; do
        echo -e "\n========================================"
        echo "开始测试：策略=$policy, 批大小=$batch_size, 速率=$arrival_rate"
        echo "惩罚因子=$penalty_factor, SLO约束=$slo_constraint, 提示数=$num_prompts"
        echo "========================================"

        # 启动服务器
        echo "启动服务器..."
        if [ "$policy" = "fcfs" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "fcfs" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --model "$model" &
        elif [ "$policy" = "concord-default-structure" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --graph-structure-type "tot" \
                --graph-matching-mode "none" \
                --model "$model" &
        elif [ "$policy" = "concord-total-deadline-no-graph" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --graph-structure-type "tot" \
                --graph-matching-mode "none" \
                --use-total-deadline \
                --model "$model" &
        elif [ "$policy" = "concord-static-default-structure" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --graph-structure-type "tot" \
                --graph-matching-mode "static" \
                --model "$model" &
        elif [ "$policy" = "concord-static-total-deadline" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --graph-structure-type "tot" \
                --graph-matching-mode "static" \
                --use-total-deadline \
                --model "$model" &
        elif [ "$policy" = "concord-online-graph" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --graph-structure-type "tot" \
                --graph-matching-mode "online" \
                --model "$model" &
        elif [ "$policy" = "concord-precise" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --disable-prediction \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --graph-structure-type "tot" \
                --graph-matching-mode "precise" \
                --model "$model" &
        else
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "$policy" \
                --enable-chunked-prefill True \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --model "$model" &
        fi
        server_pid=$!

        # 等待服务器启动
        echo "等待服务器初始化（60秒）..."
        sleep 45

        # 运行性能测试
        echo "启动客户端..."
        output_file="${output_dir}/tot_exp_burst_bs${batch_size}_rate${arrival_rate}_${policy}.log"
        python3 benchmarks/benchmark_scheduler.py \
            --model "$model" \
            --policy "$policy" \
            --arrival-rate "$arrival_rate" \
            --trace-path "benchmarks/dataset/trace/lmsys_tot.json" \
            --penalty-factor "$penalty_factor" \
            --batch-size "$batch_size" \
            --slo-constraint "$slo_constraint" \
            --burst True \
            --num-prompts "$num_prompts" > "$output_file" 2>&1

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
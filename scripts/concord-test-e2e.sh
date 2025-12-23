#!/bin/bash
# 定义测试参数组合（与注释中的参数一致）
test_cases=(
    # batch_size arrival_rate penalty_factor slo_constraint num_prompts
    "16 2 1 2,0.1,20 4000"
)

policies=("vllm" "las")
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
        if [ "$policy" = "vllm" ]; then
            python3 vllm/entrypoints/api_server.py \
                --scheduling-policy "fcfs" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --model "$model" &
        elif [ "$policy" = "vtc" ]; then
            python3 vllm/entrypoints/api_server.py \
                --scheduling-policy "vtc" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --model "$model" &
        elif [ "$policy" = "concord-precise" ]; then
            python3 vllm/entrypoints/api_server.py \
                --scheduling-policy "concord" \
                --disable-prediction \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --model "$model" &
        elif [ "$policy" = "concord-wo-graph" ]; then
            python3 vllm/entrypoints/api_server.py \
                --scheduling-policy "concord" \
                --disable-graph-matching \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --model "$model" &
        elif [ "$policy" = "concord-wo-prediction" ]; then
            python3 vllm/entrypoints/api_server.py \
                --scheduling-policy "concord" \
                --use-default-length \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --model "$model" &
        else
            python3 vllm/entrypoints/api_server.py \
                --scheduling-policy "$policy" \
                --enable-chunked-prefill True \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$batch_size" \
                --model "$model" &
        fi
        server_pid=$!

        # 等待服务器启动
        echo "等待服务器初始化（40秒）..."
        sleep 60

        # 运行性能测试
        echo "启动客户端..."
        output_file="${output_dir}/exp_burst_bs${batch_size}_rate${arrival_rate}_${policy}.log"
        #output_file="a_test.log"
        python3 benchmarks/benchmark_scheduler.py \
            --model "$model" \
            --policy "$policy" \
            --arrival-rate "$arrival_rate" \
            --trace-path "benchmarks/dataset/trace/lmsys.json" \
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
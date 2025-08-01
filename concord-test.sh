#!/bin/bash

#之前的测试参数
# bs=8, arrival_rate=1.5, penalty_factor=2, slo_constraint="1,0.1,10", num_prompts=6000
# bs=16, arrival_rate=2.5, penalty_factor=2, slo_constraint="1,0.1,10", num_prompts=6000
# bs=32, arrival_rate=4.0, penalty_factor=2, slo_constraint="1,0.1,10", num_prompts=10000

# 定义测试参数
policies=("concord" "sjf" "fcfs" "vllm")
rates=(3.0)
batch_sizes=(32)
penalty_factors=(2)
output_dir="benchmark_results"
#model="Qwen/Qwen2.5-14B-Instruct"
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


for rate in "${rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for penalty_factor in "${penalty_factors[@]}"; do
            for policy in "${policies[@]}"; do
                echo -e "\n========================================"
                echo "开始测试：策略=$policy, 速率=$rate, 批大小=$batch_size, 惩罚因子=$penalty_factor"
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
                sleep 40

                # 运行性能测试
                echo "启动客户端..."
                #output_file="${output_dir}/exp_qwen_${policy}_${rate}_${batch_size}_${penalty_factor}.log"
                output_file="${output_dir}/test_${policy}.log"
                python3 benchmarks/benchmark_scheduler.py \
                    --model "$model" \
                    --policy "$policy" \
                    --arrival-rate "$rate" \
                    --penalty-factor "$penalty_factor" \
                    --batch-size "$batch_size" \
                    --slo-constraint "1,0.1,10" \
                    --num-prompts 10 > "$output_file" 2>&1

                # 停止服务器
                echo "停止服务器..."
                kill $server_pid
                wait $server_pid 2>/dev/null

                # 清理间隔
                echo "等待系统冷却（30秒）..."
                sleep 30
            done
        done
    done
done

echo -e "\n所有测试完成！结果保存在 $output_dir/ 目录"
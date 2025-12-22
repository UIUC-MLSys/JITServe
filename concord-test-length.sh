#!/bin/bash

# 4.0 800 4/8 for Llama-3.1-8B
# 4.0 2000 16/32/64 for Llama-3.1-8B
# 10.0 800 4/8 for Qwen-2.5-14B
# 10.0 2000 4/8 for Qwen-2.5-14B
# 10.0 4000 16/32/64 for Qwen-2.5-14B

# 定义测试参数
policies=("fcfs")
rates=(10.0)
batch_sizes=(16)
penalty_factors=(100)
search_strategy="sliding_window"
top_k_selection=1
#output_dir="batch_profile_results_debug"
output_dir="length_motivation"
model="Qwen/Qwen2.5-14B-Instruct"
#model="meta-llama/Llama-3.1-8B-Instruct"
qrf_model_path="/home/exouser/qrf_model/0_qrf_lmsys_chat_llama3_8b.pkl"
qrf_tokenizer_path="/home/exouser/qrf_vectorizer/0_qrf_lmsys_chat_llama3_8b.pkl"

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
                        --search-strategy "$search_strategy" \
                        --top-k-selection "$top_k_selection" \
                        --prediction-model-path "$qrf_model_path" \
                        --prediction-tokenizer-path "$qrf_tokenizer_path" \
                        --model "$model" &
                elif [ "$policy" = "vtc" ]; then
                    python3 vllm/entrypoints/api_server.py \
                        --scheduling-policy "vtc" \
                        --enable-chunked-prefill False \
                        --penalty-factor "$penalty_factor" \
                        --max-num-seqs "$batch_size" \
                        --search-strategy "$search_strategy" \
                        --top-k-selection "$top_k_selection" \
                        --prediction-model-path "$qrf_model_path" \
                        --prediction-tokenizer-path "$qrf_tokenizer_path" \
                        --model "$model" &
                elif [ "$policy" = "concord-precise" ]; then
                    python3 vllm/entrypoints/api_server.py \
                        --scheduling-policy "concord" \
                        --disable-prediction \
                        --enable-chunked-prefill True \
                        --penalty-factor "$penalty_factor" \
                        --max-num-seqs "$batch_size" \
                        --search-strategy "$search_strategy" \
                        --top-k-selection "$top_k_selection" \
                        --prediction-model-path "$qrf_model_path" \
                        --prediction-tokenizer-path "$qrf_tokenizer_path" \
                        --model "$model" &
                else
                    python3 vllm/entrypoints/api_server.py \
                        --scheduling-policy "$policy" \
                        --enable-chunked-prefill True \
                        --penalty-factor "$penalty_factor" \
                        --max-num-seqs "$batch_size" \
                        --search-strategy "$search_strategy" \
                        --top-k-selection "$top_k_selection" \
                        --prediction-model-path "$qrf_model_path" \
                        --prediction-tokenizer-path "$qrf_tokenizer_path" \
                        --model "$model" &
                fi
                server_pid=$!

                # 等待服务器启动
                echo "等待服务器初始化（40秒）..."
                sleep 40

                # 运行性能测试
                echo "启动客户端..."
                #output_file="${output_dir}/exp_qwen_${policy}_${rate}_${batch_size}_${penalty_factor}.log"
                output_file="${output_dir}/hetero-qwen-16.log"
                python3 benchmarks/benchmark_scheduler.py \
                    --model "$model" \
                    --policy "$policy" \
                    --arrival-rate "$rate" \
                    --penalty-factor "$penalty_factor" \
                    --batch-size "$batch_size" \
                    --slo-constraint "1,0.1,10" \
                    --num-prompts 800 > "$output_file" 2>&1

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

#todo: control output length
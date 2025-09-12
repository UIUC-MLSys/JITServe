#!/bin/bash

# 定义测试参数
policies=("concord")
rates=(4.0)
batch_sizes=(16)
penalty_factors=(1)
search_strategy="sliding_window"
top_k_selection=3
#output_dir="batch_profile_results_debug"
output_dir="batch_profile_decay_llama"
#model="Qwen/Qwen2.5-14B-Instruct"
model="meta-llama/Llama-3.1-8B-Instruct"

# 输入参数
request_ratios="1.0,1,1"
num_prompts="3000"

# 创建输出目录
mkdir -p "$output_dir"

# 定义清理函数
cleanup() {
    echo -e "\n捕获中断，正在清理..."
    kill $server_pid 2>/dev/null
    kill $scheduler_pid 2>/dev/null
    kill $deepresearch_pid 2>/dev/null
    exit 1
}

# 注册中断处理
trap cleanup SIGINT

# 确定使用的trace文件和deepresearch请求数
trace_file="benchmarks/dataset/trace/lmsys.json"
num_deep_research=0
rate_deepresearch=0
rate_scheduler=0

if [[ -n "$request_ratios" && -n "$num_prompts" ]]; then
    echo "使用自定义请求比例，生成trace文件..."
    
    # 运行create_trace.py并捕获输出
    create_output=$(python3 benchmarks/create_trace.py --request_ratio "$request_ratios" --num_prompts "$num_prompts" 2>&1)
    echo "$create_output"
    
    # 解析trace文件路径和deepresearch请求数
    trace_file=$(echo "$create_output" | grep "Trace file:" | sed 's/Trace file: //')
    num_deep_research=$(echo "$create_output" | grep "Number of deepresearch requests calculated:" | sed 's/Number of deepresearch requests calculated: //')
    
    if [[ -z "$trace_file" || -z "$num_deep_research" ]]; then
        echo "错误: 无法解析create_trace.py的输出"
        echo "输出内容: $create_output"
        exit 1
    fi
    
    echo "使用trace文件: $trace_file"
    echo "Deepresearch请求数: $num_deep_research"
else
    echo "使用默认trace文件: $trace_file"
    if [[ -n "$num_prompts" ]]; then
        echo "警告: 指定了num_prompts但未指定request_ratios，将使用默认trace"
    fi
fi

for rate in "${rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for penalty_factor in "${penalty_factors[@]}"; do
            for policy in "${policies[@]}"; do
                echo -e "\n========================================"
                echo "开始测试：策略=$policy, 速率=$rate, 批大小=$batch_size, 惩罚因子=$penalty_factor"
                echo "========================================"

                # 计算到达率和请求数
                if [[ $num_deep_research -gt 0 ]]; then
                    # 计算deepresearch和scheduler的到达率
                    rate_deepresearch=$(python3 -c "print($rate * ($num_deep_research / $num_prompts))")
                    rate_scheduler=$(python3 -c "print($rate - $rate_deepresearch)")
                    # 计算lmsys请求数
                    num_lmsys=$(python3 -c "print($num_prompts - $num_deep_research)")
                    echo "Deepresearch到达率: $rate_deepresearch"
                    echo "Scheduler到达率: $rate_scheduler"
                    echo "LMSYS请求数: $num_lmsys"
                else
                    rate_scheduler=$rate
                    num_lmsys=$num_prompts
                    echo "Scheduler到达率: $rate_scheduler"
                    echo "LMSYS请求数: $num_lmsys"
                fi

                # 启动服务器
                echo "启动服务器..."
                if [ "$policy" = "vllm" ]; then
                    python3 -m vllm.entrypoints.api_server \
                        --scheduling-policy "fcfs" \
                        --enable-chunked-prefill False \
                        --penalty-factor "$penalty_factor" \
                        --max-num-seqs "$batch_size" \
                        --search-strategy "$search_strategy" \
                        --top-k-selection "$top_k_selection" \
                        --model "$model" &
                elif [ "$policy" = "vtc" ]; then
                    python3 -m vllm.entrypoints.api_server \
                        --scheduling-policy "vtc" \
                        --enable-chunked-prefill False \
                        --penalty-factor "$penalty_factor" \
                        --max-num-seqs "$batch_size" \
                        --search-strategy "$search_strategy" \
                        --top-k-selection "$top_k_selection" \
                        --model "$model" &
                elif [ "$policy" = "concord-precise" ]; then
                    python3 -m vllm.entrypoints.api_server \
                        --scheduling-policy "concord" \
                        --disable-prediction \
                        --enable-chunked-prefill True \
                        --penalty-factor "$penalty_factor" \
                        --max-num-seqs "$batch_size" \
                        --search-strategy "$search_strategy" \
                        --top-k-selection "$top_k_selection" \
                        --model "$model" &
                else
                    python3 -m vllm.entrypoints.api_server \
                        --scheduling-policy "$policy" \
                        --enable-chunked-prefill True \
                        --penalty-factor "$penalty_factor" \
                        --max-num-seqs "$batch_size" \
                        --search-strategy "$search_strategy" \
                        --top-k-selection "$top_k_selection" \
                        --model "$model" &
                fi
                server_pid=$!

                # 等待服务器启动
                echo "等待服务器初始化（40秒）..."
                sleep 40

                # 运行基准测试
                echo "启动客户端..."
                
                # benchmark_scheduler.py
                scheduler_output_file="${output_dir}/scheduler_${policy}_${rate}_${batch_size}_${penalty_factor}.log"
                if [[ -n "$request_ratios" ]]; then
                    # 使用生成的trace文件，使用num_lmsys作为num_prompts
                    python3 benchmarks/benchmark_scheduler.py \
                        --model "$model" \
                        --policy "$policy" \
                        --arrival-rate "$rate_scheduler" \
                        --penalty-factor "$penalty_factor" \
                        --batch-size "$batch_size" \
                        --slo-constraint "1,0.1,10" \
                        --num-prompts "$num_lmsys" \
                        --trace-path "$trace_file" > "$scheduler_output_file" 2>&1 &
                else
                    # 使用默认lmsys.json，需要指定num_prompts
                    prompts_arg=""
                    if [[ -n "$num_prompts" ]]; then
                        prompts_arg="--num-prompts $num_lmsys"
                    else
                        prompts_arg="--num-prompts 200"
                    fi
                    python3 benchmarks/benchmark_scheduler.py \
                        --model "$model" \
                        --policy "$policy" \
                        --arrival-rate "$rate" \
                        --penalty-factor "$penalty_factor" \
                        --batch-size "$batch_size" \
                        --slo-constraint "1,0.1,10" \
                        $prompts_arg > "$scheduler_output_file" 2>&1 &
                fi
                scheduler_pid=$!

                # benchmark_scheduler_deepresearch.py (仅当有deepresearch请求时运行)
                if [[ $num_deep_research -gt 0 ]]; then
                    deepresearch_output_file="${output_dir}/deepresearch_${policy}_${rate}_${batch_size}_${penalty_factor}.log"
                    python3 benchmarks/benchmark_scheduler_deepresearch.py \
                        --model "$model" \
                        --policy "$policy" \
                        --arrival-rate "$rate_deepresearch" \
                        --penalty-factor "$penalty_factor" \
                        --slo-constraint "1,0.1,10" \
                        --num-prompts "$num_deep_research" \
                        --trace-path "benchmarks/dataset/trace/deepresearch_llama3_maxout1024_filtered8192_test.jsonl" > "$deepresearch_output_file" 2>&1 &
                    deepresearch_pid=$!
                    
                    echo "等待两个benchmark完成..."
                    wait $scheduler_pid
                    wait $deepresearch_pid
                else
                    echo "等待scheduler benchmark完成..."
                    wait $scheduler_pid
                fi

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
#!/bin/bash

export HF_TOKEN=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

# 定义测试参数组合
test_cases=(
    # bs arrival_rate penalty_factor slo_constraint num_prompts use_all_node stage_ratio_method
    "32 0.33 1 2,0.1,20 594 false output_length"
    "32 0.33 1 2,0.1,20 594 true output_length"
    "32 0.33 1 2,0.1,20 594 false execution_time"
    "32 0.33 1 2,0.1,20 594 true execution_time"
)

policies=(
    # "fcfs"
    "concord-default-structure"
    "concord-total-deadline-no-graph"
    "concord-static-default-structure"
    "concord-static-total-deadline"
    "concord-online-graph"
    "concord-precise"
)
output_dir="benchmark_results"
model="meta-llama/Llama-3.1-8B-Instruct"
# model="meta-llama/Llama-3.2-1B-Instruct"
# trace_path="benchmarks/dataset/deepresearch_trace_filtered_8192.jsonl"
trace_path="benchmarks/dataset/test_deepresearch_llama3_maxout1024_filtered8192.jsonl"

# 创建输出目录和日志目录
mkdir -p "$output_dir"
mkdir -p "logs"

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
    read -r bs arrival_rate penalty_factor slo_constraint num_prompts use_all_node stage_ratio_method <<< "$test_case"
    
    for policy in "${policies[@]}"; do
        echo -e "\n========================================"
        echo "开始测试：策略=$policy, 批次大小=$bs, 速率=$arrival_rate"
        echo "惩罚因子=$penalty_factor, SLO约束=$slo_constraint, 提示数=$num_prompts"
        echo "全节点模式=$use_all_node, 阶段比例方法=$stage_ratio_method"
        echo "========================================"
        
        # 构建额外的参数
        extra_args=""
        if [ "$use_all_node" = "true" ]; then
            extra_args="$extra_args --use-all-node"
        fi
        extra_args="$extra_args --stage-ratio-method $stage_ratio_method"

        # 启动服务器
        echo "启动服务器..."
        
        # 构建日志文件名
        if [ "$use_all_node" = "true" ]; then
            node_type="allnode"
        else
            node_type="supernode"
        fi
        server_log_file="logs/server_rate${arrival_rate}_${policy}_${node_type}_${stage_ratio_method}.log"
        
        if [ "$policy" = "fcfs" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "fcfs" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$bs" \
                --model "$model" > "$server_log_file" 2>&1 &
        elif [ "$policy" = "concord-default-structure" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$bs" \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "none" \
                $extra_args \
                --model "$model" > "$server_log_file" 2>&1 &
        elif [ "$policy" = "concord-total-deadline-no-graph" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$bs" \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "none" \
                --use-total-deadline \
                $extra_args \
                --model "$model" > "$server_log_file" 2>&1 &
        elif [ "$policy" = "concord-static-default-structure" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$bs" \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "static" \
                $extra_args \
                --model "$model" > "$server_log_file" 2>&1 &
        elif [ "$policy" = "concord-static-total-deadline" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$bs" \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "static" \
                --use-total-deadline \
                $extra_args \
                --model "$model" > "$server_log_file" 2>&1 &
        elif [ "$policy" = "concord-online-graph" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$bs" \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "online" \
                $extra_args \
                --model "$model" > "$server_log_file" 2>&1 &
        elif [ "$policy" = "concord-precise" ]; then
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "concord" \
                --disable-prediction \
                --enable-chunked-prefill False \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$bs" \
                --graph-structure-type "deepresearch" \
                --graph-matching-mode "precise" \
                $extra_args \
                --model "$model" > "$server_log_file" 2>&1 &
        else
            python3 -m vllm.entrypoints.api_server \
                --scheduling-policy "$policy" \
                --enable-chunked-prefill True \
                --penalty-factor "$penalty_factor" \
                --max-num-seqs "$bs" \
                --model "$model" > "$server_log_file" 2>&1 &
        fi
        server_pid=$!

        # 等待服务器启动
        echo "服务器日志文件: $server_log_file"
        echo "等待服务器初始化（60秒）..."
        sleep 60

        # 运行性能测试
        echo "启动客户端..."
        
        # 构建文件名中的节点类型标识
        if [ "$use_all_node" = "true" ]; then
            node_type="allnode"
        else
            node_type="supernode"
        fi
        
        output_file="${output_dir}/deepresearch_rate${arrival_rate}_${policy}_${node_type}_${stage_ratio_method}.log"
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
            --metric-percentiles "1,25,50,90,95,99" \
            --show-id-length \
            --is-stream False > "$output_file" 2>&1

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
import matplotlib.pyplot as plt
import numpy as np

# 配置全局样式
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelpad': 10,
    'grid.alpha': 0.3
})
#Llama3.1-8B BS=16 3000
#Qwen2.5-14B BS=16 4000

# 实验数据（示例数据）
models = {
    "Llama-3.1-8B-Instruct": {
        "Concord+Density": {
            "rps": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "slo": [85.3, 79.9, 75.7, 71.5, 68.9, 64.5]
        },
        "Concord+SJF": {
            "rps": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "slo": [67.5, 59.4, 53.3, 51.2, 50.2, 50.9]
        },
        "Sarathi-Serve(FCFS)": {
            "rps": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "slo": [82.9, 75.2, 74.5, 57.8, 57.0, 36.1]
        },
        "vLLM": {
            "rps": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "slo": [70.4, 65.4, 57.8, 40.2, 34.8, 15.0]
        },
        "Autellix(LAS)": {
            "rps": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "slo": [43.4, 34.1, 13.2, 9.3, 10.2, 10.1]
        }
    },
    "Qwen2.5-14B-Instruct": {
        "Concord+Density": {
            "rps": [1.2, 1.5, 1.6, 1.7, 1.8],
            "slo": [82.7, 69.6, 64.1, 60.8, 59.9]
        },
        "Concord+SJF": {
            "rps": [1.2, 1.5, 1.6, 1.7, 1.8],
            "slo": [82.7, 47.3, 45.7, 43.6, 39.5]
        },
        "Sarathi-Serve(FCFS)": {
            "rps": [1.2, 1.5, 1.6, 1.7, 1.8],
            "slo": [71.3, 50.2, 33.5, 12.5, 6.5]
        },
        "vLLM": {
            "rps": [1.2, 1.5, 1.6, 1.7, 1.8],
            "slo": [67.2, 18.2, 4.8, 4.2, 4.5]
        },
        "Autellix(LAS)": {
            "rps": [1.2, 1.5, 1.6, 1.7, 1.8],
            "slo": [54.1, 5.3, 5.2, 4.7, 5.6]
        }
    }
}

# 创建可视化布局
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
fig.suptitle("SLO Attainment Analysis Across Models and Policies", y=1.02, fontsize=16)

#['#E6B47C', '#F0BF78', '#969696', 'dimgray', 'lightgray']
# 颜色和样式配置
policy_styles = {
    "Concord+Density": {'color': '#E6B47C', 'marker': 'o', 'ls': '-'},
    "Sarathi-Serve(FCFS)": {'color': '#969696', 'marker': 's', 'ls': '--'},
    "vLLM": {'color': 'dimgray', 'marker': 'x', 'ls': '-'},
    "Autellix(LAS)": {'color': 'lightgray', 'marker': 'D', 'ls': '-.'},
    "Concord+SJF": {'color': '#F0BF78', 'marker': '^', 'ls': ':'}
}

# 绘制每个模型的子图
for idx, (model_name, model_data) in enumerate(models.items()):
    ax = axs[idx]
    
    # 绘制每个策略的曲线
    for policy_name, policy_data in model_data.items():
        style = policy_styles[policy_name]
        ax.plot(policy_data["rps"], 
                policy_data["slo"],
                label=policy_name,
                **style,
                markersize=8,
                linewidth=2.5,
                alpha=0.9)
    
    # 子图装饰
    ax.set_title(f"{model_name}", pad=12)
    ax.set_xlabel("Requests Per Second (RPS)", labelpad=10)
    ax.set_ylabel("SLO Attainment (%)" if idx == 0 else "")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 100)
    

    for policy_name in model_data:
        data = model_data[policy_name]
        for x, y in zip(data["rps"], data["slo"]):
            ax.text(x, y+2, f'{y}%', 
                   ha='center', 
                   fontsize=9,
                   color=policy_styles[policy_name]['color'])

# 统一图例
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, 
           loc='upper center', 
           ncol=5, 
           bbox_to_anchor=(0.5, 1.0),
           frameon=False,
           fontsize=12)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85, wspace=0.08)
plt.savefig('ablation_rps.png', dpi=300, bbox_inches='tight')
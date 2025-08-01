import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch

# Llama3.1-8B BS=16, RPS=2.3 3000

# 假设的数据结构示例
data = {
    "TTFT": {
        "Concord+Density": {"p50": 0.39, "p95": 26.46},
        "Concord+SJF": {"p50": 0.44, "p95": 140.49},
        "Sarathi-Serve(FCFS)": {"p50": 2.28, "p95": 11.54},
        "vLLM": {"p50": 5.77, "p95": 19.09},
        "Autellix(LAS)": {"p50": 54.53, "p95": 105.80}
    },
    "TBT": {
        "Concord+Density": {"p50": 22.0, "p95": 30.7},
        "Concord+SJF": {"p50": 23.3, "p95": 52.5},
        "Sarathi-Serve(FCFS)": {"p50": 21.6, "p95": 29.6},
        "vLLM": {"p50": 22.0, "p95": 26.9},
        "Autellix(LAS)": {"p50": 22.3, "p95": 44.7}
    },
    "Throughput_TTLT": {
        "Concord+Density": {"p50": 9.03, "p95": 31.56},
        "Concord+SJF": {"p50": 12.03, "p95": 428.42},
        "Sarathi-Serve(FCFS)": {"p50": 9.71, "p95": 24.64},
        "vLLM": {"p50": 13.35, "p95": 30.24},
        "Autellix(LAS)": {"p50": 75.03, "p95": 214.70}
    },
    # 可能需要我将其替换成task_TTLT
    "Collective_TTLT": {
        "Concord+Density": {"p50": 9.00, "p95": 123.94},
        "Concord+SJF": {"p50": 8.17, "p95": 346.94},
        "Sarathi-Serve(FCFS)": {"p50": 9.62, "p95": 26.32},
        "vLLM": {"p50": 14.23, "p95": 33.77},
        "Autellix(LAS)": {"p50": 120.22, "p95": 332.75}
    }
}

# 颜色配置
colors = {
    'Concord+Density': '#E6B47C',
    'Concord+SJF': '#F0BF78',
    'Sarathi-Serve(FCFS)': '#969696',
    'vLLM': 'dimgray',
    'Autellix(LAS)': 'lightgray'
}

# 创建1x4的子图布局
fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=100)

# 方案一：对数坐标轴
def plot_log_scale(ax, metric_data, title):
    strategies = list(metric_data.keys())
    x = np.arange(len(strategies))
    width = 0.35
    
    # 绘制p50和p95柱状图
    p50_bars = ax.bar(x - width/2, 
                     [v['p50'] for v in metric_data.values()], 
                     width, 
                     color=[colors[s] for s in strategies],
                     alpha=0.7,
                     label='p50')
    
    p95_bars = ax.bar(x + width/2, 
                     [v['p95'] for v in metric_data.values()], 
                     width, 
                     color=[colors[s] for s in strategies],
                     alpha=0.9,
                     hatch='//',
                     label='p95')
    
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    
    # 添加数值标签
    for bars in [p50_bars, p95_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=9)
    
    # 创建自定义图例
    legend_elements = [
        Patch(facecolor='white', edgecolor='gray', alpha=0.7, label='p50'),
        Patch(facecolor='white', edgecolor='gray', alpha=0.7, hatch='//', label='p95')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# 选择绘制方案
plot_func = plot_log_scale

# 绘制四个子图
metrics = [
    ('TTFT', 'Latency-sensitive TTFT(s)'),
    ('TBT', 'Latency-sensitive TBT(ms)'),
    ('Throughput_TTLT', 'Throughput-intensive TTLT(s)'),
    ('Collective_TTLT', 'Collective TTLT(s)')
]

for idx, (metric_key, title) in enumerate(metrics):
    plot_func(axes[idx], data[metric_key], title)

# 调整整体布局
plt.tight_layout(pad=3.0)
plt.subplots_adjust(top=0.85)  # 为总标题留空间

# 添加总标题
fig.suptitle('Metrics Breakdown', 
             fontsize=16, y=0.98)
# 保存图像
plt.savefig('breakdown_metrics.png', 
            bbox_inches='tight', 
            dpi=300,
            transparent=False)
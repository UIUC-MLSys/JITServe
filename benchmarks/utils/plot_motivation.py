import matplotlib.pyplot as plt
import numpy as np

# Model: Llama3.1-8B-Instruct
# Rate: 2.6
# BS: 16
# penalty_factor: 1
# slo: 2,0.1,20

strategies = {
    "SJF": {
        "p99_TBT": 113.6,
        "p99_TTLT": 800.6,
        "SLO_Attainment": 49.5,
    },
    "Autellix": {
        "p99_TBT": 86.6,
        "p99_TTLT": 494.9,
        "SLO_Attainment": 8.6
    },
    "Sarathi-serve": {
        "p99_TBT": 42.8,
        "p99_TTLT": 53.6,
        "SLO_Attainment": 21.4
    },
}

# 可视化配置
colors = ['#969696', 'dimgray', 'lightgray']  # 策略颜色
bar_width = 0.1                   # 调整宽度
metrics = ['p99_TBT', 'p99_TTLT', 'SLO_Attainment']
titles = [
    'P99 TBT (ms)',
    'P99 TTLT (s)',
    'SLO Attainment Rate (%)'
]

# 创建画布和子图
fig, axs = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'wspace': 0.3})
#fig.suptitle('Performance Comparison: SJF vs Sarathi-serve', y=1.02, fontsize=16)

# 遍历每个指标绘制子图
for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axs[idx]
    
    # 设置x轴定位点（1/3和2/3位置）
    x_pos = np.array([0.167, 0.5, 0.833])
    
    # 提取数据
    values = [strategies[policy][metric] for policy in strategies]
    
    # 绘制柱状图
    bars = ax.bar(x_pos, values, width=bar_width, color=colors)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, 
                height + 0.02*max(values),
                f'{height:.1f}' if metric != 'SLO_Attainment' else f'{int(height)}%',
                ha='center', 
                va='bottom',
                fontsize=20)
    
    # 装饰子图
    ax.set_xlim(0, 1)  # 固定x轴范围
    ax.set_xticks(x_pos)
    #ax.set_xticklabels(strategies.keys(), fontsize=20, rotation=45)
    ax.set_ylabel(title, fontsize=20)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 设置Y轴范围
    y_max = max(values) * 1.2
    ax.set_ylim(0, y_max if metric != 'SLO_Attainment' else 100)
    ax.tick_params(axis='y', labelsize=20)
    if metric == 'SLO_Attainment':
        ax.set_yticks(np.arange(0, 101, 20))

# 统一图例
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
fig.legend(handles, strategies.keys(),
          loc='lower center',
          ncol=3,
          frameon=False,
          #bbox_to_anchor=(0.5, 1.0),
          fontsize=20)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('solution_limitation.png', dpi=300, bbox_inches='tight')
import matplotlib.pyplot as plt
import numpy as np

# ---------- 数据配置 ----------
# 任务类型 (第7题的行标签)
tasks = [
    "Real-time Chat",
    "Code Generation",
    "Batch Data Processing",
    "Paper Editing",
    "Long-Document Q&A",
]

# 指标名称 (第7题的列标签)
metrics = ["Immediacy", "Coherence", "Intervenability", "Efficiency", "Completeness"]

# 用户认为重要的比例数据 (第7题的数据按行提取)
data = np.array([
    [20+72, 10+52, 4+21, 10+47, 14+52],    # 实时对话/会议记录
    [23+4, 31+6, 51+12, 22+8, 83+25],    # 生成项目代码
    [22+4, 24+5, 31+9, 53+13, 76+20],    # 批量数据处理/模型渲染
    [27+6, 29+8, 59+11, 25+7, 79+22],     # 迭代修改论文
    [8, 13, 8, 7, 20],    # 长文档问答
])

# 将数据转换为百分比
# 将数据转换为百分比
percent_data = data / data.sum(axis=1, keepdims=True) * 100

data = percent_data

# ---------- 绘图参数 ----------

colors = ['#1f77b4', '#ff7f0e', '#ffbb78', '#006400', '#2ca02c']
bar_width = 0.15
x = np.arange(len(tasks))  # 任务标签位置

# ---------- 绘图 ----------
fig, ax = plt.subplots(figsize=(12, 6))

# 为每个metric绘制柱状图
for i in range(len(metrics)):
    ax.bar(
        x + i*bar_width, 
        data[:, i], 
        width=bar_width, 
        color=colors[i], 
        label=metrics[i]
    )

# 标签与美化
ax.set_ylabel('Percentage of Users (%)', fontsize=12)
ax.set_title('User Priorities Across Task Types and SLO Metrics', fontsize=14)
ax.set_xticks(x + bar_width*2)
ax.set_xticklabels(tasks, rotation=15, ha='right', fontsize=10)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 图例放在右侧
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('survey_results.png', dpi=300)  # 保存图像
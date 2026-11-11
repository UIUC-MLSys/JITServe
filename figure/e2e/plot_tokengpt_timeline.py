import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import re

from pathlib import Path
from typing import Dict, List, Optional
from ..utils import parse_timeline

import matplotlib.pyplot as plt
import numpy as np


def plot_service_gain_comparison(
    policies: dict,
    model: str,
    ax: plt.Axes,
    time_window_seconds: int = 60,
    gain_type: str = "Total",
    interpolate_missing: bool = False,
    start_time_minutes: Optional[float] = None,
    end_time_minutes: Optional[float] = None,
    highlight_policy: Optional[str] = None,
    alpha: float = 0.2,
    moving_avg_window: Optional[int] = None
) -> None:
    time_window_minutes = time_window_seconds / 60
    
    all_times = set()
    for policy_data in policies.values():
        all_times.update(t / 60 for t in policy_data.keys())
    all_times = sorted(all_times)

    start_time = start_time_minutes or min(all_times, default=0)
    end_time = end_time_minutes or max(all_times, default=0)
    # filtered_times = [t for t in all_times if start_time <= t <= end_time]

    highlight_color = None
    highlight_values = []
    all_moving_times = []

    for policy_name, policy_data in policies.items():
        sorted_data = sorted(
            [(t / 60, gains) for t, gains in policy_data.items()],
            key=lambda x: x[0]
        )
        times = [t for t, _ in sorted_data if start_time <= t <= end_time]
        gains = [g for t, g in sorted_data if start_time <= t <= end_time]

        if interpolate_missing:
            filled_gains = []
            current_idx = 0

            # handle missing values
            for t in times:
                if current_idx < len(times) and t == times[current_idx]:
                    filled_gains.append(gains[current_idx])
                    current_idx += 1
                else:
                    if current_idx == 0:
                        filled_gains.append(0)
                    else:
                        if current_idx < len(times):
                            t_prev = times[current_idx - 1]
                            t_next = times[current_idx]
                            y_prev = gains[current_idx - 1]
                            y_next = gains[current_idx]
                            delta = (t - t_prev) / (t_next - t_prev)
                            filled_gains.append(y_prev + delta * (y_next - y_prev))
                        else:
                            filled_gains.append(gains[-1])
            y_values = filled_gains
            x_values = times
        else:
            y_values = gains
            x_values = times

        # compute moving avg
        if moving_avg_window:
            window_size = max(int(moving_avg_window / time_window_minutes), 1)
            if len(y_values) >= window_size:
                moving_avg = np.convolve(y_values, np.ones(window_size) / window_size, mode='valid')
                moving_times = x_values[window_size - 1:]
            else:
                print(f"Warning: Not enough data points for moving average ({policy_name})")
                moving_avg = None
                moving_times = None
        else:
            moving_avg = None
            moving_times = None

        if moving_avg is not None and moving_times is not None:
            all_moving_times.extend(moving_times)
            line, = ax.plot(
                moving_times, moving_avg,
                linestyle='-' if policy_name not in ["LTR", "vLLM"] else '--',
                linewidth=2,
                label=f"{policy_name})"
            )

        if policy_name == highlight_policy:
            highlight_color = line.get_color()
            highlight_values = list(zip(times, gains))

    # if highlight_policy and highlight_color and highlight_values:
    #     h_times, h_gains = zip(*highlight_values)
    #     y_min = np.min(h_gains)
    #     y_max = np.max(h_gains)
    #     
    #     ax.fill_betweenx(
    #         y=[y_min, y_max],
    #         x1=start_time,
    #         x2=end_time,
    #         color=highlight_color,
    #         alpha=alpha
    #     )

    ax.set_title(f"{model}", fontsize=20, pad=10)
    if model == "Llama-3.1-8B-Instruct":
        ax.set_ylabel("Token Goodput (token/s)", fontsize=20)
    ax.grid(True, linestyle='-' if policy_name not in ["LTR", "vLLM"] else '--', alpha=0.5)
    
    ax.xaxis.set_major_locator(ticker.FixedLocator(ax.get_xticks()))
    ax.set_xticklabels(
        [f"{int(x)}" for x in ax.get_xticks()],
        ha='right'
    )

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major', labelsize=15)

    if moving_avg_window and all_moving_times:
        window_start = min(all_moving_times)
        window_end = max(all_moving_times)
        time_margin = (window_end - window_start) * 0.05
        ax.set_xlim(
            16,
            window_end + time_margin
        )
    else:
        ax.set_xlim(start_time, end_time)

    all_y_data = []
    for line in ax.get_lines():
        y_data = line.get_ydata()
        if len(y_data) > 0:
            all_y_data.append(np.max(y_data))
    
    if all_y_data:
        current_max = max(all_y_data)
        ax.set_ylim(0, current_max * 1.1)
    else:
        ax.set_ylim(0, 100)


def plot(models, directory):
    fig, axes = plt.subplots(1, len(models), figsize=(35, 7), sharex=False, squeeze=False)
    axes = axes.flatten()
    plt.subplots_adjust(top=0.80, hspace=3)

    common_params = {
        "time_window_seconds": 360,
        "gain_type": "Total",
        "interpolate_missing": True,
        "start_time_minutes": 0,
        "end_time_minutes": 80,
        "highlight_policy": "Concord+Density",
        "alpha": 0.15,
        "moving_avg_window": 60
    }

    for idx, model in enumerate(models):
        data_dir = Path(directory) / model

        files = os.listdir(data_dir)
        experiments = {}
        for f in files:
            if not f.endswith(".log"):
                continue

            match = re.match(r"(scheduler|deepresearch)_([^_]+)_[\d.]+_", f)
            if match:
                prefix, strategy = match.groups()
                key = strategy
                experiments.setdefault(key, {})
                if prefix == "scheduler":
                    experiments[key]['normal'] = os.path.join(data_dir, f)
                else:
                    experiments[key]['deep'] = os.path.join(data_dir, f)

        results = {}
        for key, paths in experiments.items():
            if 'normal' in paths and 'deep' in paths:
                token_goodput = parse_timeline(paths['normal'], paths['deep'], ['token_goodput'])
                results[key] = token_goodput['token_goodput']

        policies = {
            "JITServe": results.get("jitserve"),
            "LTR": results.get("ltr"),
            "vLLM": results.get("vllm"),
            "Sarathi-serve": results.get("fcfs"),
            "Autellix": results.get("autellix"),
        }

        plot_service_gain_comparison(
            policies=policies,
            model=model,
            ax=axes[idx],
            **common_params
        )

    handles, labels = axes[0].get_legend_handles_labels()
    labels = [label[:-1] for label in labels]

    fig.legend(handles, labels, 
               loc='upper center', 
               ncol=len(labels),
               bbox_to_anchor=(0.5, 0.93),
               fontsize=15,
               frameon=False)
    fig.text(0.5, 0, 'Time (minutes)', ha='center', fontsize=20)

    plt.savefig(
        "figure/e2e/e2e_token_goodput_timeline.pdf",
        bbox_inches='tight',
        dpi=300
    )
    plt.close()


if __name__ == "__main__":
    models = ["Llama-3.1-8B-Instruct"]
    plot(models, "batch_result/e2e-timeline")
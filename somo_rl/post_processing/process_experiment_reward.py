import os
import sys
import argparse
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, path)

from user_settings import EXPERIMENT_ABS_PATH
from somo_rl.post_processing.process_run_reward import Process_reward_data


def plot_run_data(ax, x, y_df, color, linewidth=1, std_df=None):
    y = np.array(y_df)
    std = np.array(std_df) if std_df is not None else None

    ax.plot(x[:len(y)], y, color=color, linewidth=linewidth)[0]
    if std is not None:
        ax.fill_between(x[:len(y)], y - std, y + std, color=color, alpha=0.1)
                
def smooth_df_cols(df, window_size):
    smoothed_df = pd.DataFrame()
    for col in df:
        data = df[col].to_numpy()
        data = (np.convolve(data, np.ones(window_size), 'full') / window_size)
        smoothed_df[col] = data
    return smoothed_df

class Process_experiment:
    def __init__(self, exp_abs_path, exp_name, run_group_names=(), run_IDs=()):
        # run_group_names = [f"obs-{i}" for i in [1, 3, 6, 10, 20]]
        run_group_names = list(run_group_names)
        run_IDs = list(run_IDs)
        self.exp_dir = Path(exp_abs_path) / exp_name
        if len(run_IDs) == 0:
            run_group_names = run_group_names if len(run_group_names) != 0 else os.listdir(self.exp_dir)
            for group in run_group_names:
                for run in os.listdir(self.exp_dir / group):
                    run_IDs.append([exp_name, group, run])

        self.run_rewards = []
        self.run_IDs = []
        fail_count = 0
        for run_ID in run_IDs:
            try:
                run_reward_data = Process_reward_data(exp_abs_path=exp_abs_path, run_ID=run_ID)
                run_reward_data.process_monitor_data()
                self.run_rewards.append(run_reward_data)
                self.run_IDs.append(run_ID)
            except:
                print(f"FAILED PARSING RUN: {run_ID}")
                fail_count += 1
        print(f"ALL RUNS PARSED. RUN PARSING FAIL COUNT: {fail_count}")

        # self.exp_results_dir = self.exp_dir / "exp_results"


    def plot_runs(
        self,
        title="",
        ylabel="",
        log_component="r",
        x_units="episodes",
        max_x_val=None,
        smoothing=1,
        ref_line_data=None,
        ref_line_name="No Action Taken",
        show=True,
        # save_figs=True,
    ):

        _fig, ax = plt.subplots(1, 1)

        for i, run in enumerate(self.run_rewards):
            run.plot(
                ax,
                title=title,
                ylabel=f"{log_component} value",
                label=str(run.run_ID),
                log_component=log_component,
                x_units=x_units,
                max_x_val=max_x_val,
                smoothed=(smoothing != 1),
                ref_line_data=ref_line_data,
                ref_line_name=ref_line_name
            )

        plt.title(title if title else "Reward History Overlay")

        # if save_figs:
        #     plt.savefig(self.exp_results_dir / (title.replace(" ", "_") + ".png"))
        #     plt.savefig(
        #         self.exp_results_dir / (title.replace(" ", "_") + ".eps"), format="eps"
        #     )

        if show:
            ax.legend()
            plt.show()


    def plot_grouped_runs(
        self,
        processing_groups={},
        title="Reward History Overlay",
        ylabel="",
        log_component="r",
        max_x_val=None,
        smoothing=1,
        ref_line_data=None,
        ref_line_name="No Action Taken",
        plot_all_runs=False,
        plot_mean=True,
        show=True,
        show_std = True
        # save_figs=True,
    ):
        x_units = "episodes"
        _fig, ax = plt.subplots(1, 1)

        if len(processing_groups) == 0:
            # default processing_groups to run_groups
            run_group_names = sorted(list(set(np.array(self.run_IDs)[:,1])))
            # run_group_nums = sorted([int(name.split("-")[1]) for name in list(set(np.array(self.run_IDs)[:,1]))])
            # run_group_names = [f"net-{num}" for num in run_group_nums]
            for run_group in run_group_names:
                processing_groups[run_group] = []
            for i, run_ID in enumerate(self.run_IDs):
                processing_groups[run_ID[1]].append(self.run_rewards[i])
        self.processing_groups = processing_groups

        cNorm  = colors.Normalize(vmin=0, vmax=(len(processing_groups) - 1))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('jet'))
        legend_elements = [None] * len(processing_groups)
        longest_x = []

        for i, group in enumerate(self.processing_groups):
            group_rewards_df = pd.DataFrame()
            group_xs_df = pd.DataFrame()
            for run in self.processing_groups[group]:
                group_rewards_df[str(run.run_ID)] = deepcopy(run.monitoring_means_df[log_component])
                group_xs_df[str(run.run_ID)] = pd.Series(run.get_x_data("episodes"))
            
            # x = np.array(group_xs_df.mean(axis=1))[int((smoothing/2) - 1): int(-(smoothing/2))]
            smoothed_rewards_df = smooth_df_cols(group_rewards_df, smoothing)
            x = np.arange(0, len(smoothed_rewards_df))

            if len(x) > len(longest_x):
                longest_x = deepcopy(x)

            color = scalarMap.to_rgba(i)
            
            if plot_all_runs:
                for j, y_df in enumerate([smoothed_rewards_df[col] for col in smoothed_rewards_df]):
                    plot_run_data(ax, x, y_df, color, linewidth=0.25)
            if (not plot_all_runs) or (plot_all_runs and plot_mean):
                means = smoothed_rewards_df.mean(axis=1)[:-smoothing]
                stds = smoothed_rewards_df.std(axis=1)[:-smoothing]
                if not show_std:
                    stds = None
                plot_run_data(ax, x[:-smoothing], means, color, std_df=stds, linewidth=1.25)

            legend_elements[i] = Line2D([0], [0], color=color, lw=4, label=group)

        if ref_line_data is not None:
            ax.plot(longest_x, [ref_line_data] * len(longest_x), "k--", label=ref_line_name)
            legend_elements.append(Line2D([0], [0], linestyle="--", color='black', label=ref_line_name))
                
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel if ylabel != "" else f"{log_component} value")
        ax.set_xlabel(x_units)
        ax.legend(handles=legend_elements)

        # if save_figs:
        #     plt.savefig(self.exp_results_dir / (title.replace(" ", "_") + ".png"))
        #     plt.savefig(
        #         self.exp_results_dir / (title.replace(" ", "_") + ".eps"), format="eps"
        #     )

        if show:
            plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args to quickly run experiment postprocessing")
    parser.add_argument(
        "-e",
        "--exp_name",
        help="Experiment name",
        required=True,
        default=None,
    )
    parser.add_argument(
        "--exp_abs_path",
        help="Experiment directory absolute path",
        required=False,
        default=EXPERIMENT_ABS_PATH,
    )
    parser.add_argument(
        "-lc",
        "--log_component",
        help="Component of monitor log to plot",
        required=False,
        default="r",
    )
    parser.add_argument(
        "-sg",
        "--show_grouped",
        help="divide runs into groups",
        action="store_true",
    )
    parser.add_argument(
        "--smoothing",
        help="window size for smoothing",
        required=False,
        default=1
    )
    parser.add_argument(
        "--no_std",
        help="don't show standard deviation",
        action="store_true",
    )
    parser.add_argument(
        "--plot_all_runs",
        help="plot a line for each run",
        action="store_true",
    )
    arg = parser.parse_args()
    exp_data = Process_experiment(
        arg.exp_abs_path,
        arg.exp_name
    )
    if arg.show_grouped:
        exp_data.plot_grouped_runs(plot_all_runs=arg.plot_all_runs, log_component=arg.log_component, smoothing=int(arg.smoothing), show_std = not arg.no_std)
    else:
        exp_data.plot_grouped_runs(smoothing=int(arg.smoothing))


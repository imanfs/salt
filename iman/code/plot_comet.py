import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from comet_ml import API


def plot_style():
    # Update multiple rcParams at once
    plt.rcParams.update({
        "figure.dpi": 144,  # Set figure DPI
        "font.size": 10,  # Set global font size
        "xtick.labelsize": "medium",  # Set x-tick label size
        "ytick.labelsize": "medium",  # Set y-tick label size
        "xtick.major.size": 4,  # Set x-tick major size
        "xtick.major.width": 1,  # Set x-tick major width
        "ytick.major.size": 4,  # Set y-tick major size
        "ytick.major.width": 1,  # Set y-tick major width
        "axes.linewidth": 1,  # Set axes line width
        "axes.labelsize": "medium",  # Set axes label size
        "axes.titlesize": "medium",  # Set axes title size
        "legend.fontsize": 8,  # Set axes legend size
    })


def format_plot(name=None, show_axis=False, fig=None, ax=None):
    plot_style()
    if show_axis:
        ax.axhline(color="k", lw=0.5)
        ax.axvline(color="k", lw=0.5)
    if name is not None:
        if fig is not None:
            fig.savefig(name, bbox_inches="tight", dpi=300)
        else:
            print("Figure object not provided, saving with plt.savefig.")
            plt.savefig(name, bbox_inches="tight", dpi=300)


class Plotter:
    def __init__(self):
        self.task_names = [
            "jets_classification",
            "regression",
            "mask_ce",
            "mask_dice",
            "track_origin",
            "object_class_ce",
        ]
        self.colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])[:6]
        self.task_colors = dict(zip(self.task_names, self.colors, strict=False))

    def plot_cossim(
        self, metric_name, metric_data, main_task_name, smoothing_name, ax=None, metric_type=None
    ):
        if ax is None:
            fig, ax = plt.subplots()

        self.clean_name(metric_name, main_task_name, metric_type=metric_type)
        metric_label = self.metric_label
        ax.plot(
            metric_data["step"],
            metric_data[smoothing_name],
            label=self.task_label,
            color=self.task_colors[self.task_cleaned_name],
            lw=0.75,
            alpha=0.8,
        )
        ax.set_xlabel("Step")
        ax.set_ylabel(self.metric_label)
        other_tasks = (
            " of " + self.main_task_name + " with other tasks"
            if self.metric_type == "cos_sim"
            else ""
        )
        title = metric_label + other_tasks
        ax.set_title(title)
        ax.legend()

        if ax is None:
            return fig, ax
        return ax

    def split_modify_case_combine(self, metric_name, combine_str=" "):
        words = metric_name.split("_")
        if combine_str == " ":
            words = [
                word.upper() if word in {"ce", "dice"} else word.capitalize() for word in words
            ]
        return combine_str.join(words)

    def clean_name(self, metric, main_task=None, metric_type=None):
        self.metric_type = metric_type
        # If the metric type is given and present in the string, extract and remove it
        if metric_type and metric_type in metric:
            metric_cleaned = metric.replace(metric_type + "_", "").replace("_" + metric_type, "")
            self.metric_label = self.split_modify_case_combine(metric_type)
        else:
            metric_cleaned = metric  # If no specific metric_type is given

        if main_task and main_task in metric_cleaned:
            metric_cleaned = metric_cleaned.replace(main_task + "_", "").replace(
                "_" + main_task, ""
            )

        self.main_task_name = self.split_modify_case_combine(main_task)
        self.task_label = self.split_modify_case_combine(metric_cleaned)
        self.task_cleaned_name = self.split_modify_case_combine(metric_cleaned, combine_str="_")


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode="same")


api_key = "xwLSnGIZdSJEBPjktmQTZiBv0"
api = API(api_key=api_key)

# Replace with your workspace, project name, and experiment key
workspace = "isanai"
project_name = "salt"
experiment_key = "d3ff503afa724b39b8712dc63abad83b"

experiment = api.get_experiment(
    workspace=workspace, project_name=project_name, experiment=experiment_key
)

metrics = experiment.get_metrics()

# Create a DataFrame from the metrics
metrics_df = pd.DataFrame(metrics)
metrics_df["metricValue"] = metrics_df["metricValue"].astype(float)
window = 35
metrics_df["smoothedValue"] = metrics_df["metricValue"].rolling(window=window).mean()
metrics_df["EMAValue"] = metrics_df["metricValue"].ewm(span=5, adjust=False).mean()

plotter = Plotter()

fig_el1_grads, ax_el1_grads = plt.subplots()
fig_el2_grads, ax_el2_grads = plt.subplots()

ax_el1_grads.axvline(x=12000, color="k", linestyle="--", lw=0.5, alpha=0.5)
ax_el2_grads.axvline(x=12000, color="k", linestyle="--", lw=0.5, alpha=0.5)

plot_dir = "/home/xucabis2/salt/iman/plots/figs/"

for task in plotter.task_names:
    fig_cos_sim, ax_cos_sim = plt.subplots()
    ax_cos_sim.axvline(x=12000, color="k", linestyle="--", lw=0.5, alpha=0.5)
    for metric in metrics_df["metricName"].unique():
        if "cos_sim" in metric and task in metric:
            window = 35
            metrics_df["smoothedValue"] = metrics_df["metricValue"].rolling(window=window).mean()
            metric_data = metrics_df[metrics_df["metricName"] == metric]
            plotter.plot_cossim(
                metric,
                metric_data[window:],
                task,
                "smoothedValue",
                ax=ax_cos_sim,
                metric_type="cos_sim",
            )
            plot_name_cossim = "cos_sim_" + task + ".png"
        if "grad_L1" in metric and task in metric:
            window = 15
            metrics_df["smoothedValue"] = metrics_df["metricValue"].rolling(window=window).mean()
            metric_data = metrics_df[metrics_df["metricName"] == metric]
            plotter.plot_cossim(
                metric,
                metric_data[window:],
                task,
                "smoothedValue",
                ax=ax_el1_grads,
                metric_type="grad_L1",
            )
            plot_name_el1 = plotter.metric_type + ".png"
        if "grad_L2" in metric and task in metric:
            window = 15
            metrics_df["smoothedValue"] = metrics_df["metricValue"].rolling(window=window).mean()
            metric_data = metrics_df[metrics_df["metricName"] == metric]
            plotter.plot_cossim(
                metric,
                metric_data[window:],
                task,
                "smoothedValue",
                ax=ax_el2_grads,
                metric_type="grad_L2",
            )
            plot_name_el2 = plotter.metric_type + ".png"
    format_plot(plot_dir + plot_name_cossim, show_axis=True, fig=fig_cos_sim, ax=ax_cos_sim)
format_plot(plot_dir + plot_name_el1, show_axis=True, fig=fig_el1_grads, ax=ax_el1_grads)
format_plot(plot_dir + plot_name_el2, show_axis=True, fig=fig_el2_grads, ax=ax_el2_grads)

# -*- coding: utf-8 -*-
# file: core.py
# time: 12:43 2023/1/25
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.

import datetime
import json
import os
import pickle
import random
from collections import OrderedDict

import findfile
import matplotlib
import natsort
import pandas as pd
from findfile import find_cwd_files
import numpy as np
from scipy.stats import ranksums
from tabulate import tabulate

from metric_visualizer import __version__ as version
from metric_visualizer import __name__ as pkg_name
from metric_visualizer.utils import MetricList

mv_font = {
    "family": "Serif",
    "weight": "normal",
    "size": 15,
}

mv_font_small = {
    "family": "Serif",
    "weight": "normal",
    "size": 10,
}

mv_font_large = {
    "family": "Serif",
    "weight": "normal",
    "size": 20,
}

# set font for matplotlib
matplotlib.rc("font", **mv_font)

tex_template = r"""
    \documentclass{article}
    \usepackage{pgfplots}
    \usepackage{tikz}
    \usepackage{caption}
    \usetikzlibrary{intersections}
    \usepackage{helvet}
    \usepackage[eulergreek]{sansmath}
    \usepackage{amsfonts,amssymb,amsmath,amsthm,amsopn}	% math related
    \usetikzlibrary{patterns}
    \begin{document}

        \begin{figure}
        \centering

        $tikz_code$

        \end{figure}

    \end{document}
    """


class MetricVisualizer:
    COLORS_DICT = matplotlib.colors.XKCD_COLORS
    COLORS_DICT.update(matplotlib.colors.CSS4_COLORS)
    COLORS = list(COLORS_DICT.values())
    MARKERS = [
        ".",
        "o",
        "+",
        "P",
        "x",
        "X",
        "D",
        "d",
    ]

    HATCHES = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

    def __init__(self, name, *, metric_dict=None, **kwargs):
        self.name = name
        self.version = version
        self.pkg_name = pkg_name

        if metric_dict is None:
            self.metrics = OrderedDict(
                {
                    # Example data:
                    # 'Accuracy': {
                    #     'trial0': [77.06, 2.52, 35.14, 3.04, 77.29, 3.8, 57.4, 38.52, 60.36, 22.45],
                    #     'train1': [64.53, 58.33, 97.89, 68.12, 88.6, 60.33, 70.99, 75.91, 42.49, 15.03],
                    #     'train2': [97.74, 86.05, 41.34, 81.66, 75.08, 1.76, 94.63, 27.26, 47.11, 42.06],
                    # },
                    # 'Macro-F1': {
                    #     'trial0': [111.5, 105.61, 179.08, 167.25, 181.85, 152.75, 194.82, 130.86, 108.51, 151.44],
                    #     'train1': [187.58, 106.35, 134.22, 167.68, 188.24, 196.54, 154.21, 193.71, 183.34, 150.18],
                    #     'train2': [159.24, 148.44, 119.49, 160.24, 169.6, 133.27, 129.36, 180.36, 165.24, 152.38],
                    # }
                }
            )
        else:
            self.metrics = metric_dict

        self.trial2unit = {}

        self.trial_rank_test_result = {}
        self.metric_rank_test_result = {}

    @staticmethod
    def compile_tikz(crop=True, clean=True, **kwargs):
        for f in findfile.find_cwd_files(
            ".tex", exclude_key=["ignore", ".pdf"], recursive=kwargs.get("recursive", 1)
        ):
            os.system(f"pdflatex {f} {f}.pdf")
        # for f in findfile.find_cwd_files(".tex", exclude_key=["ignore", ".pdf"]):
        #     os.system(f"rm {f}")
        if crop:
            for f in findfile.find_cwd_files(".pdf", exclude_key=["ignore", ".pdf"]):
                os.system(f"pdfcrop {f} {f}")
        if clean:
            for f in findfile.find_cwd_files(".aux", exclude_key=["ignore", ".pdf"]):
                os.remove(f)
            for f in findfile.find_cwd_files(".log", exclude_key=["ignore", ".pdf"]):
                os.remove(f)
            for f in findfile.find_cwd_files(".out", exclude_key=["ignore", ".pdf"]):
                os.remove(f)

    def log(self, trial_name=None, metric_name=None, value=0, unit=None):
        """
        Add a metric to the metric dict based on the trial name and metric name.
        :param trial_name: the name of the trial, such as algo names, model names, config names, epochs, etc.
        :param metric_name: the name of the metric, such as accuracy, loss, f1, etc.
        :param value: the value of the metric
        :param unit: the unit of the metric, such as %, ms, etc.

        :return: None
        """
        self.log_metric(trial_name, metric_name, value, unit)

    def log_metric(self, trial_name=None, metric_name=None, value=0, unit=None):
        """
        Add a metric to the metric dict based on the trial name and metric name.
        :param trial_name: the name of the trial, such as algo names, model names, config names, epochs, etc.
        :param metric_name: the name of the metric, such as accuracy, loss, f1, etc.
        :param value: the value of the metric
        :param unit: the unit of the metric, such as %, ms, etc.

        :return: None
        """
        assert metric_name is not None

        # if unit is not None, add the unit to the trial name
        self.trial2unit[metric_name] = unit

        # if trial_name is None, use the length of the metric dict as the trial name
        if trial_name is None:
            trial_name = "Trial{}".format(
                len(self.metrics[metric_name]) + 1 if metric_name in self.metrics else 1
            )

        # add the metric to the metric dict
        if metric_name in self.metrics:
            if trial_name not in self.metrics[metric_name]:
                self.metrics[metric_name][trial_name] = MetricList([value])
            else:
                self.metrics[metric_name][trial_name].append(value)
        else:
            self.metrics[metric_name] = {trial_name: MetricList([value])}
        return self

    def set_trial_names(self, trial_names):
        """
        Set the trial names.
        :param trial_names: the trial names

        :return: None
        """
        for metric_name in self.metrics:
            self.metrics[metric_name] = OrderedDict(
                zip(trial_names, self.metrics[metric_name].values())
            )

    def set_trial_colors(self, trial_colors):
        """
        Set the trial colors.
        :param trial_colors: the trial colors

        :return: None
        """
        for metric_name in self.metrics:
            for trial_name, trial_color in zip(self.metrics[metric_name], trial_colors):
                self.metrics[metric_name][trial_name].color = trial_color

    def set_metric_names(self, metric_names):
        """
        Set the metric names.
        :param metric_names: the metric names

        :return: None
        """
        self.metrics = OrderedDict(zip(metric_names, self.metrics.values()))

    def set_metric_colors(self, metric_colors):
        """
        Set the metric colors.
        :param metric_colors: the metric colors

        :return: None
        """
        for metric_name, metric_color in zip(self.metrics, metric_colors):
            for trial_name in self.metrics[metric_name]:
                self.metrics[metric_name][trial_name].color = metric_color

    def box_plot(
        self, by="trial", engine="matplotlib", save_path=None, show=True, **kwargs
    ):
        """
        Draw a box plot based on the metric name and trial name.
        :param by: the name of the x-axis, such as trial, metric, etc.
        :param engine: the engine to draw the box plot, such as matplotlib, tikz, etc.
        :param save_path: the path to save the box plot
        :param show: whether to show the box plot

        :return: None
        """
        import matplotlib.pyplot as plt

        plt.cla()

        if by == "trial":
            plot_metrics = self.metrics
        else:
            plot_metrics = self.transpose()

        width = kwargs.pop("widths", 0.9)

        # get the number of metrics
        num_metrics = len(plot_metrics.keys())
        # get the number of trials
        num_trials = len(plot_metrics[list(plot_metrics.keys())[0]].keys())

        # get the width of the box plot
        width = width / num_metrics if kwargs.get("no_overlap", True) else width

        # get the xtick labels
        xtick_labels = list(plot_metrics[list(plot_metrics.keys())[0]].keys())
        # get the xticks
        xticks = (
            np.arange(num_trials) + width * num_metrics
            if kwargs.get("no_overlap", True)
            else np.arange(num_trials) + width / 2
        )

        # get the colors
        colors = plt.cm.jet(np.linspace(0, 1, num_metrics))
        box_parts = []
        # draw the box plot
        ax = plt.subplot()
        for i, metric_name in enumerate(plot_metrics.keys()):
            # get the values
            values = list(plot_metrics[metric_name].values())
            # draw the box plot
            box_part = ax.boxplot(
                values,
                labels=xtick_labels,
                positions=xticks + i * width
                if kwargs.get("no_overlap", True)
                else xticks,
                widths=width * 0.9,
                # patch_artist=kwargs.get("patch_artist", True),
                boxprops=dict(linewidth=2, color=colors[i]),
                capprops=dict(linewidth=2, color=colors[i]),
                whiskerprops=dict(linewidth=2, color=colors[i]),
                flierprops=dict(
                    linewidth=2, color=colors[i], markeredgecolor=colors[i]
                ),
                medianprops=dict(linewidth=2, color=colors[i]),
                **kwargs.get("boxplot_kwargs", {}),
            )

            box_parts.append(box_part["boxes"][0])

            for item in ["boxes", "whiskers", "fliers", "medians", "caps"]:
                plt.setp(box_part[item], color=colors[i])

            plt.setp(box_part["fliers"], markeredgecolor=colors[i])

        if kwargs.get("legend", True):
            ax.legend(
                box_parts, list(plot_metrics.keys()), loc=kwargs.pop("legend_loc", 1)
            )

        if kwargs.get("minor_ticks", True):
            ax.minorticks_on()

        if kwargs.get("grid", True):
            ax.grid(which="major", linestyle="-", linewidth="0.3", color="grey")
            ax.grid(which="minor", linestyle=":", linewidth="0.3", color="grey")

        plt.xticks(
            kwargs.get(
                "xticks",
                xticks + i / 2 * width if kwargs.get("no_overlap", True) else xticks,
            ),
            xtick_labels
            if not kwargs.get("xticklabels", None)
            else kwargs.get("xticklabels"),
            rotation=kwargs.pop("xrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            # verticalalignment=kwargs.pop("verticalalignment", "top"),
            **kwargs.pop("xticks_kwargs", {}),
        )
        plt.yticks(
            rotation=kwargs.pop("yrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "right"),
            verticalalignment=kwargs.pop("verticalalignment", "center"),
            **kwargs.pop("yticks_kwargs", {}),
        )

        plt.xlabel(kwargs.pop("xlabel", "Trial Name"))
        plt.ylabel(kwargs.pop("ylabel", "Metric Value"))

        if kwargs.get("tight_layout", True):
            plt.tight_layout()

        # if kwargs.get("xticklabels", True):
        #     ax.set_xticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         # verticalalignment=kwargs.pop("verticalalignment", "top"),
        #         **kwargs.pop("xticklabels_kwargs", {}),
        #     )
        # if kwargs.get("yticklabels", True):
        #     ax.set_yticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         verticalalignment=kwargs.pop("verticalalignment", "baseline"),
        #         **kwargs.pop("yticklabels_kwargs", {}),
        #     )
        if engine != "tikz":
            # save the box plot
            if save_path is not None:
                plt.savefig(save_path, dpi=kwargs.pop("dpi", 1000))
            # show the box plot
            if show:
                plt.show()
        else:
            import tikzplotlib

            tex_code = tikzplotlib.get_tikz_code()
            tex_code = tex_template.replace("$tikz_code$", tex_code)
            if save_path is not None:
                with open(save_path, "w") as f:
                    f.write(tex_code)
            if show:
                print(tex_code)

            return save_path

    def violin_plot(
        self, by="trial", engine="matplotlib", save_path=None, show=True, **kwargs
    ):
        """
        Draw a violin plot based on the metric name and trial name.
        :param by: the name of the x-axis, such as trial, metric, etc.
        :param engine: the engine to draw the violin plot, such as matplotlib, tikz, etc.
        :param save_path: the path to save the violin plot
        :param show: whether to show the violin plot

        :return: None
        """
        import matplotlib.pyplot as plt

        plt.cla()

        if by == "trial":
            plot_metrics = self.metrics
        else:
            plot_metrics = self.transpose()

        width = kwargs.pop("width", 0.9)

        # get the number of metrics
        num_metrics = len(plot_metrics.keys())
        # get the number of trials
        num_trials = len(plot_metrics[list(plot_metrics.keys())[0]].keys())

        # get the width of the violin plot
        width = width / num_metrics if kwargs.get("no_overlap", True) else width
        # get the xticks
        xticks = (
            np.arange(num_trials) + width / 2
            if kwargs.get("no_overlap", True)
            else np.arange(num_trials)
        )
        xtick_labels = list(plot_metrics[list(plot_metrics.keys())[0]].keys())
        # get the xtick labels

        violin_parts = []
        # draw the violin plot
        ax = plt.subplot()
        for i, metric_name in enumerate(plot_metrics.keys()):
            # get the values
            values = list(plot_metrics[metric_name].values())
            # draw the violin plot
            violin = ax.violinplot(
                values,
                positions=xticks + i * width
                if kwargs.get("no_overlap", True)
                else xticks,
                widths=width * 0.9,
                showmeans=kwargs.get("showmeans", False),
                showmedians=kwargs.get("showmedians", True),
                showextrema=kwargs.get("showextrema", True),
                bw_method=kwargs.get("bw_method", "scott"),
                **kwargs.get("violinplot_kwargs", {}),
            )

            violin_parts.append(violin["bodies"][0])

        if kwargs.get("legend", True):
            plt.legend(
                violin_parts, plot_metrics.keys(), loc=kwargs.pop("legend_loc", 1)
            )

        for pc in violin["bodies"]:
            pc.set_linewidth(kwargs.pop("linewidth", 3))

        if kwargs.get("minor_ticks", True):
            plt.minorticks_on()

        if kwargs.get("grid", True):
            plt.grid(which="major", linestyle="-", linewidth="0.3", color="grey")
            plt.grid(which="minor", linestyle=":", linewidth="0.3", color="grey")

        plt.xticks(
            kwargs.get(
                "xticks",
                xticks + i * width / 2 if kwargs.get("no_overlap", True) else xticks,
            ),
            xtick_labels
            if not kwargs.get("xticklabels", None)
            else kwargs.get("xticklabels"),
            rotation=kwargs.pop("xrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            # verticalalignment=kwargs.pop("verticalalignment", "top"),
            **kwargs.pop("xticks_kwargs", {}),
        )
        plt.yticks(
            rotation=kwargs.pop("yrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "right"),
            verticalalignment=kwargs.pop("verticalalignment", "center"),
            **kwargs.pop("yticks_kwargs", {}),
        )

        plt.xlabel(kwargs.pop("xlabel", "Trial Name"))
        plt.ylabel(kwargs.pop("ylabel", "Metric Value"))

        if kwargs.get("tight_layout", True):
            plt.tight_layout()

        # if kwargs.get("xticklabels", True):
        #     ax.set_xticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         # verticalalignment=kwargs.pop("verticalalignment", "top"),
        #         **kwargs.pop("xticklabels_kwargs", {}),
        #     )
        # if kwargs.get("yticklabels", True):
        #     ax.set_yticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         verticalalignment=kwargs.pop("verticalalignment", "baseline"),
        #         **kwargs.pop("yticklabels_kwargs", {}),
        #     )
        if engine != "tikz":
            # save the violin plot
            if save_path is not None:
                plt.savefig(save_path, dpi=kwargs.pop("dpi", 1000))
            # show the violin plot
            if show:
                plt.show()
        else:
            import tikzplotlib

            tex_code = tikzplotlib.get_tikz_code()
            tex_code = tex_template.replace("$tikz_code$", tex_code)
            if save_path is not None:
                with open(save_path, "w") as f:
                    f.write(tex_code)
            if show:
                print(tex_code)

            return save_path

    def pie_plot(
        self, by="trial", engine="matplotlib", save_path=None, show=True, **kwargs
    ):
        """
        Draw a pie plot based on the metric name and trial name.
        :param by: the name of the x-axis, such as trial, metric, etc.
        :param engine: the engine to draw the pie plot, such as matplotlib, tikz, etc.
        :param save_path: the path to save the pie plot
        :param show: whether to show the pie plot

        :return: None
        """
        import matplotlib.pyplot as plt

        plt.cla()
        if by != "trial":
            plot_metrics = self.transpose()
        else:
            plot_metrics = self.metrics

        # get the number of metrics
        num_metrics = len(plot_metrics.keys())
        # get the number of trials
        num_trials = len(plot_metrics[list(plot_metrics.keys())[0]])
        # get the xticks
        xticks = np.arange(num_metrics)
        xtick_labels = list(plot_metrics.keys())
        # get the xtick labels
        # get the yticks
        yticks = np.arange(num_trials)
        ytick_labels = list(plot_metrics[list(plot_metrics.keys())[0]].keys())
        # get the ytick labels

        # draw the pie plot
        ax = plt.subplot()
        for i, metric_name in enumerate(plot_metrics.keys()):
            # get the values
            values = list(plot_metrics[metric_name].values())
            # draw the pie plot
            pie = ax.pie(
                [np.nanmean(vec) for vec in values],
                labels=ytick_labels,
                center=kwargs.pop("center", (0, 0)),
                **kwargs.pop("pieplot_kwargs", {}),
            )

        # if kwargs.get("xticklabels", True):
        #     ax.set_xticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         # verticalalignment=kwargs.pop("verticalalignment", "top"),
        #         **kwargs.pop("xticklabels_kwargs", {}),
        #     )
        # if kwargs.get("yticklabels", True):
        #     ax.set_yticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         verticalalignment=kwargs.pop("verticalalignment", "baseline"),
        #         **kwargs.pop("yticklabels_kwargs", {}),
        #     )
        if engine != "tikz":
            # save the violin plot
            if save_path is not None:
                plt.savefig(save_path, dpi=kwargs.pop("dpi", 1000))
            # show the violin plot
            if show:
                plt.show()
        else:
            import tikzplotlib

            tex_code = tikzplotlib.get_tikz_code()
            tex_code = tex_template.replace("$tikz_code$", tex_code)
            if save_path is not None:
                with open(save_path, "w") as f:
                    f.write(tex_code)
            if show:
                print(tex_code)

            return save_path

    def scatter_plot(
        self, by="trial", engine="matplotlib", save_path=None, show=True, **kwargs
    ):
        """
        Draw a scatter plot based on the metric name and trial name.
        :param by: the name of the x-axis, such as trial, metric, etc.
        :param engine: the engine to draw the scatter plot, such as matplotlib, tikz, etc.
        :param save_path: the path to save the scatter plot
        :param show: whether to show the scatter plot

        :return: None
        """
        import matplotlib.pyplot as plt

        plt.cla()
        if by != "trial":
            plot_metrics = self.transpose()
        else:
            plot_metrics = self.metrics

            # get the number of metrics
        num_metrics = len(plot_metrics.keys())
        # get the number of trials
        num_trials = len(plot_metrics[list(plot_metrics.keys())[0]].keys())

        # get the width of the scatter plot
        width = 0.8 / num_metrics
        # get the xticks
        xticks = np.arange(num_trials) + 0.4
        # get the xtick labels
        xtick_labels = list(plot_metrics[list(plot_metrics.keys())[0]].keys())

        # get the colors
        colors = plt.cm.jet(np.linspace(0, 1, num_metrics))

        # draw the scatter plot
        ax = plt.subplot()
        scatter_parts = []
        for i, metric_name in enumerate(plot_metrics.keys()):
            # get the values
            values = list(plot_metrics[metric_name].values())
            # draw the scatter plot
            scatter_part = ax.scatter(
                [[x] * len(values[0]) for x in (xticks + i * width)],
                values,
                color=colors[i],
                **kwargs.get("scatter_kwargs", {}),
            )

            scatter_parts.append(scatter_part)

        if kwargs.get("legend", True):
            plt.legend(
                scatter_parts, plot_metrics.keys(), loc=kwargs.pop("legend_loc", 1)
            )

        if kwargs.get("minor_ticks", True):
            plt.minorticks_on()

        if kwargs.get("grid", True):
            plt.grid(which="major", linestyle="-", linewidth="0.3", color="grey")
            plt.grid(which="minor", linestyle=":", linewidth="0.3", color="grey")

        plt.xticks(
            kwargs.get(
                "xticks",
                xticks + i * width / 2 if kwargs.get("no_overlap", True) else xticks,
            ),
            xtick_labels
            if not kwargs.get("xtick_labels", None)
            else kwargs.get("xtick_labels", None),
            rotation=kwargs.pop("xrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            # verticalalignment=kwargs.pop("verticalalignment", "top"),
            **kwargs.pop("xticks_kwargs", {}),
        )
        plt.yticks(
            rotation=kwargs.pop("yrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "right"),
            verticalalignment=kwargs.pop("verticalalignment", "center"),
            **kwargs.pop("yticks_kwargs", {}),
        )

        plt.xlabel(kwargs.pop("xlabel", "Trial Name"))
        plt.ylabel(kwargs.pop("ylabel", "Metric Value"))

        if kwargs.get("tight_layout", True):
            plt.tight_layout()

        # if kwargs.get("xticklabels", True):
        #     ax.set_xticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         # verticalalignment=kwargs.pop("verticalalignment", "top"),
        #         **kwargs.pop("xticklabels_kwargs", {}),
        #     )
        # if kwargs.get("yticklabels", True):
        #     ax.set_yticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         verticalalignment=kwargs.pop("verticalalignment", "baseline"),
        #         **kwargs.pop("yticklabels_kwargs", {}),
        #     )
        if engine != "tikz":
            # save the scatter plot
            if save_path is not None:
                plt.savefig(save_path, dpi=kwargs.pop("dpi", 1000))
            # show the scatter plot
            if show:
                plt.show()
        else:
            import tikzplotlib

            tex_code = tikzplotlib.get_tikz_code()
            tex_code = tex_template.replace("$tikz_code$", tex_code)
            if save_path is not None:
                with open(save_path, "w") as f:
                    f.write(tex_code)
            if show:
                print(tex_code)

            return save_path

    def trajectory_plot(
        self, by="trial", engine="matplotlib", save_path=None, show=True, **kwargs
    ):
        """
        Draw a trajectory plot based on the metric name and trial name.
        :param by: the name of the x-axis, such as trial, metric, etc.
        :param engine: the engine to draw the trajectory plot, such as matplotlib, tikz, etc.
        :param save_path: the path to save the trajectory plot
        :param show: whether to show the trajectory plot

        :return: None
        """
        import matplotlib.pyplot as plt

        plt.cla()

        if by == "trial":
            plot_metrics = self.metrics
        else:
            plot_metrics = self.transpose()

        if not kwargs.get("markers", None):
            markers = self.MARKERS[:]
        else:
            markers = kwargs.pop("markers")

        if not kwargs.get("colors", None):
            colors = self.COLORS[:]
        else:
            colors = kwargs.pop("colors")

        traj_parts = []
        ax = plt.subplot()

        for metric_name in plot_metrics.keys():
            metrics = plot_metrics[metric_name]

            y = np.array([metrics[metric_name] for metric_name in metrics])
            x = np.array(
                [[i for i, label in enumerate(metrics)] for _ in range(y.shape[1])]
            )

            y_avg = np.average(y, axis=1)
            y_std = np.std(y, axis=1)

            marker = random.choice(markers)
            markers.remove(marker)
            if not colors:
                colors = self.COLORS[:]
            color = random.choice(colors)
            colors.remove(color)

            if kwargs.pop("avg_point", True):
                avg_point = ax.plot(
                    x[0],
                    y_avg,
                    marker=marker,
                    color=color,
                    markersize=kwargs.pop("markersize", 3),
                    linewidth=kwargs.pop("linewidth", 3),
                )

            if kwargs.pop("traj_fill", True):
                plt.subplot().fill_between(
                    x[0],
                    y_avg - y_std,
                    y_avg + y_std,
                    color=color,
                    alpha=kwargs.pop("alpha", 0.2),
                )

            if kwargs.pop("traj_point", True):
                plt.subplot().scatter(x, y, marker=marker, color=color)

            traj_parts.append(avg_point[0])

        if kwargs.get("legend", True):
            plt.legend(
                traj_parts, list(plot_metrics.keys()), loc=kwargs.pop("legend_loc", 1)
            )

        if kwargs.get("minor_ticks", True):
            plt.minorticks_on()

        if kwargs.get("grid", True):
            plt.grid(which="major", linestyle="-", linewidth="0.3", color="grey")
            plt.grid(which="minor", linestyle=":", linewidth="0.3", color="grey")

        plt.xticks(
            kwargs.get("xticks", list(range(len(metrics.keys())))),
            list(metrics.keys())
            if not kwargs.get("xticklabels", None)
            else kwargs.get("xticklabels", None),
            rotation=kwargs.pop("xrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            **kwargs.pop("xticks_kwargs", {}),
        )
        plt.yticks(
            rotation=kwargs.pop("yrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "right"),
            verticalalignment=kwargs.pop("verticalalignment", "center"),
            **kwargs.pop("yticks_kwargs", {}),
        )

        plt.xlabel(kwargs.pop("xlabel", "Trial Name"))
        plt.ylabel(kwargs.pop("ylabel", "Metric Value"))

        if kwargs.get("tight_layout", True):
            plt.tight_layout()

        # if kwargs.get("xticklabels", True):
        #     ax.set_xticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         # verticalalignment=kwargs.pop("verticalalignment", "top"),
        #         **kwargs.pop("xticklabels_kwargs", {}),
        #     )
        # if kwargs.get("yticklabels", True):
        #     ax.set_yticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         verticalalignment=kwargs.pop("verticalalignment", "baseline"),
        #         **kwargs.pop("yticklabels_kwargs", {}),
        #     )
        if engine != "tikz":
            # save the trajectory plot
            if save_path is not None:
                plt.savefig(save_path, dpi=kwargs.pop("dpi", 1000))
            # show the trajectory plot
            if show:
                plt.show()
        else:
            import tikzplotlib

            tex_code = tikzplotlib.get_tikz_code()
            tex_code = tex_template.replace("$tikz_code$", tex_code)
            if save_path is not None:
                with open(save_path, "w") as f:
                    f.write(tex_code)
            if show:
                print(tex_code)

            return save_path

    def bar_plot(
        self, by="trial", engine="matplotlib", save_path=None, show=True, **kwargs
    ):
        """
        Draw a bar plot based on the metric name and trial name.
        :param by: the name of the x-axis, such as trial, metric, etc.
        :param engine: the engine to draw the bar plot, such as matplotlib, tikz, etc.
        :param save_path: the path to save the bar plot
        :param show: whether to show the bar plot

        :return: None
        """
        import matplotlib.pyplot as plt

        plt.cla()

        if by == "trial":
            plot_metrics = self.metrics
        else:
            plot_metrics = self.transpose()

        if not kwargs.get("markers", None):
            markers = self.MARKERS[:]
        else:
            markers = kwargs.pop("markers")

        if not kwargs.get("colors", None):
            colors = self.COLORS[:]
        else:
            colors = kwargs.pop("colors")

        bar_parts = []
        ax = plt.subplot()
        total_width = 0.9
        for i, metric_name in enumerate(plot_metrics.keys()):
            metrics = plot_metrics[metric_name]
            metric_num = len(plot_metrics.keys())
            trial_num = len(plot_metrics[metric_name])
            width = total_width / metric_num
            x = np.arange(trial_num)
            x = x - (total_width - width) / 2
            x = x + i * width
            Y = np.array(
                [
                    np.average(plot_metrics[m_name][trial])
                    for m_name in plot_metrics.keys()
                    for trial in plot_metrics[m_name]
                    if metric_name == m_name
                ]
            )
            hatch = random.choice(self.HATCHES)
            color = random.choice(colors)
            colors.remove(color)
            bar = plt.bar(x, Y, width=width, hatch=hatch, color=color)
            bar_parts.append(bar[0])

            for i_x, j_x in zip(x, Y):
                plt.text(
                    i_x, j_x + max(Y) // 100, "%.1f" % j_x, ha="center", va="bottom"
                )

        if kwargs.get("legend", True):
            plt.legend(
                bar_parts, list(plot_metrics.keys()), loc=kwargs.pop("legend_loc", 1)
            )

        if kwargs.get("minor_ticks", True):
            plt.minorticks_on()

        if kwargs.get("grid", True):
            plt.grid(which="major", linestyle="-", linewidth="0.3", color="grey")
            plt.grid(which="minor", linestyle=":", linewidth="0.3", color="grey")

        plt.xticks(
            kwargs.get("xticks", list(range(len(metrics.keys())))),
            list(metrics.keys()),
            rotation=kwargs.pop("xrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            # verticalalignment=kwargs.pop("verticalalignment", "top"),
            **kwargs.pop("xticks_kwargs", {}),
        )
        plt.yticks(
            rotation=kwargs.pop("yrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "right"),
            verticalalignment=kwargs.pop("verticalalignment", "center"),
            **kwargs.pop("yticks_kwargs", {}),
        )

        plt.xlabel(kwargs.pop("xlabel", "Trial Name"))
        plt.ylabel(kwargs.pop("ylabel", "Metric Value"))

        if kwargs.get("tight_layout", True):
            plt.tight_layout()

        # if kwargs.get("xticklabels", True):
        #     ax.set_xticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         # verticalalignment=kwargs.pop("verticalalignment", "top"),
        #         **kwargs.pop("xticklabels_kwargs", {}),
        #     )
        # if kwargs.get("yticklabels", True):
        #     ax.set_yticklabels(
        #         [metric_name for metric_name in metrics],
        #         rotation=kwargs.pop("rotation", 0),
        #         rotation_mode=kwargs.pop("rotation_mode", "anchor"),
        #         horizontalalignment=kwargs.pop("horizontalalignment", "center"),
        #         verticalalignment=kwargs.pop("verticalalignment", "baseline"),
        #         **kwargs.pop("yticklabels_kwargs", {}),
        #     )

        if engine != "tikz":
            # save the bar plot
            if save_path is not None:
                plt.savefig(save_path, dpi=kwargs.pop("dpi", 1000))
            # show the bar plot
            if show:
                plt.show()
        else:
            import tikzplotlib

            tex_code = tikzplotlib.get_tikz_code()
            tex_code = tex_template.replace("$tikz_code$", tex_code)
            if save_path is not None:
                with open(save_path, "w") as f:
                    f.write(tex_code)
            if show:
                print(tex_code)

            return save_path

    def a12_bar_plot(
        self,
        target_trial=None,
        engine="matplotlib",
        save_path=None,
        show=True,
        **kwargs,
    ):
        """
        Draw a bar plot based on the metric name and trial name.
        :param target_trial:  the target trial to compare with other trials
        :param engine: the engine to draw the bar plot, such as matplotlib, tikz, etc.
        :param save_path: the path to save the bar plot
        :param show: whether to show the bar plot

        :return: None
        """

        try:
            from rpy2 import robjects
            from rpy2.robjects import pandas2ri
        except ImportError:
            raise ImportError(
                "You need to \n1): install R programming language (https://cran.r-project.org/mirrors.html)."
                '\n2): install "effsize" by R prompt: \ninstall.packages("effsize")'
                "\n3): pip install rpy2\n"
            )

        pandas2ri.activate()
        r_cmd = """
                require(effsize)

                method1<-c($data1$)
                method2<-c($data2$)

                categs <- rep(c("method1", "method2"), each=$num$)
                VD.A(c(method1,method2), categs)

                """
        plot_metrics = self.transpose()
        if target_trial is None:
            new_plot_metrics = {
                "large": {trial: [0] for trial in plot_metrics.keys()},
                "medium": {trial: [0] for trial in plot_metrics.keys()},
                "small": {trial: [0] for trial in plot_metrics.keys()},
                "equal": {trial: [0] for trial in plot_metrics.keys()},
            }
            max_num = 0
            count = 0
            for trial1 in plot_metrics.keys():
                for metric in plot_metrics[trial1].keys():
                    for trial2 in plot_metrics.keys():
                        if trial1 != trial2:
                            cmd = r_cmd.replace(
                                "$data1$",
                                ", ".join(
                                    natsort.natsorted(
                                        [str(x) for x in plot_metrics[trial1][metric]]
                                    )
                                ),
                            )
                            cmd = cmd.replace(
                                "$data2$",
                                ", ".join(
                                    natsort.natsorted(
                                        [str(x) for x in plot_metrics[trial2][metric]]
                                    )
                                ),
                            )
                            cmd = cmd.replace(
                                "$num$", str(len(plot_metrics[trial1][metric]))
                            )
                            res = robjects.r(cmd)

                            if "large" in str(res):
                                new_plot_metrics["large"][trial1][0] += 1
                            elif "medium" in str(res):
                                new_plot_metrics["medium"][trial1][0] += 1
                            elif "small" in str(res):
                                new_plot_metrics["small"][trial1][0] += 1
                            elif "equal" in str(res):
                                new_plot_metrics["equal"][trial1][0] += 1
                            elif "negligible" in str(res):
                                new_plot_metrics["equal"][trial1][0] += 1
                            else:
                                print(res)
                                raise RuntimeError("Unknown Error")
                            max_num = max(
                                max_num,
                                new_plot_metrics["large"][trial1][0],
                                new_plot_metrics["medium"][trial1][0],
                                new_plot_metrics["small"][trial1][0],
                                new_plot_metrics["equal"][trial1][0],
                            )
                            count += 1
            count /= len(plot_metrics.keys())
            for metric in new_plot_metrics.keys():
                for trial in new_plot_metrics[metric].keys():
                    new_plot_metrics[metric][trial][0] = max(
                        round(new_plot_metrics[metric][trial][0] / count * 100, 2),
                        new_plot_metrics[metric][trial][0] / count * 5,
                    )
            plot_metrics = new_plot_metrics

        elif target_trial >= 0:
            new_plot_metrics = {
                "large": {
                    trial: [0]
                    for trial in list(plot_metrics.keys())[:target_trial]
                    + list(plot_metrics.keys())[target_trial + 1 :]
                },
                "medium": {
                    trial: [0]
                    for trial in list(plot_metrics.keys())[:target_trial]
                    + list(plot_metrics.keys())[target_trial + 1 :]
                },
                "small": {
                    trial: [0]
                    for trial in list(plot_metrics.keys())[:target_trial]
                    + list(plot_metrics.keys())[target_trial + 1 :]
                },
                "equal": {
                    trial: [0]
                    for trial in list(plot_metrics.keys())[:target_trial]
                    + list(plot_metrics.keys())[target_trial + 1 :]
                },
            }
            max_num = 0
            count = 0
            trial1 = list(plot_metrics.keys())[target_trial]
            for trial2 in (
                list(plot_metrics.keys())[:target_trial]
                + list(plot_metrics.keys())[target_trial + 1 :]
            ):
                for metric in plot_metrics[trial2].keys():
                    cmd = r_cmd.replace(
                        "$data1$",
                        ", ".join(
                            natsort.natsorted(
                                [str(x) for x in plot_metrics[trial1][metric]]
                            )
                        ),
                    )
                    cmd = cmd.replace(
                        "$data2$",
                        ", ".join(
                            natsort.natsorted(
                                [str(x) for x in plot_metrics[trial2][metric]]
                            )
                        ),
                    )
                    cmd = cmd.replace("$num$", str(len(plot_metrics[trial1][metric])))
                    res = robjects.r(cmd)

                    if "large" in str(res):
                        new_plot_metrics["large"][trial2][0] += 1
                    elif "medium" in str(res):
                        new_plot_metrics["medium"][trial2][0] += 1
                    elif "small" in str(res):
                        new_plot_metrics["small"][trial2][0] += 1
                    elif "equal" in str(res):
                        new_plot_metrics["equal"][trial2][0] += 1
                    elif "negligible" in str(res):
                        new_plot_metrics["equal"][trial2][0] += 1
                    else:
                        print(res)
                        raise RuntimeError("Unknown Error")
                    max_num = max(
                        max_num,
                        new_plot_metrics["large"][trial2][0],
                        new_plot_metrics["medium"][trial2][0],
                        new_plot_metrics["small"][trial2][0],
                        new_plot_metrics["equal"][trial2][0],
                    )
                    count += 1
            count /= len(plot_metrics.keys()) - 1
            for metric in new_plot_metrics.keys():
                for trial in new_plot_metrics[metric].keys():
                    new_plot_metrics[metric][trial][0] = max(
                        round(new_plot_metrics[metric][trial][0] / count * 100, 2),
                        new_plot_metrics[metric][trial][0] / count * 5,
                    )
            plot_metrics = new_plot_metrics

        mv = MetricVisualizer(name=self.name + " A12", metrics=plot_metrics)
        return mv.bar_plot(
            by="trial", engine=engine, save_path=save_path, show=show, **kwargs
        )

    def sk_rank_plot(
        self, plot_type="box", engine="matplotlib", save_path=None, show=True, **kwargs
    ):
        """
        Draw a rank plot based on the metric name and trial name.
        :param plot_type: the type of the rank plot, such as box, violin, etc.
        :param engine: the engine to draw the bar plot, such as matplotlib, tikz, etc.
        :param save_path: the path to save the bar plot
        :param show: whether to show the bar plot

        :return: None
        """

        from metric_visualizer.external import Rx

        # metrics = self.transpose()
        metrics = self.metrics

        Rx.list_algorithm_rank = []
        data_dict = {"Scott-Knott Rank Test": {}}

        for metric in metrics:
            Rx.data(**metrics[metric])
            for i, d in enumerate(natsort.natsorted(Rx.list_algorithm_rank)):
                data_dict["Scott-Knott Rank Test"][d[0]] = data_dict[
                    "Scott-Knott Rank Test"
                ].get(d[0], [])
                data_dict["Scott-Knott Rank Test"][d[0]].append(d[1])

        mv = MetricVisualizer(
            name=self.name + ".sk_rank",
            metric_dict=data_dict,
        )
        if plot_type == "box":
            return mv.box_plot(
                engine=engine,
                save_path=save_path,
                show=show,
                ylabel="Scott-Knott Rank",
                **kwargs,
            )
        else:
            return mv.violin_plot(
                engine=engine,
                save_path=save_path,
                show=show,
                ylabel="Scott-Knott Rank",
                **kwargs,
            )

    def remove_outliers(self, outlier_constant=1.5):
        """Remove outliers from the data.

        Args:
            outlier_constant (float, optional): The constant used to calculate the
                upper and lower bounds. Defaults to 1.5.
        """
        for metric_name, metric_data in self.metrics.items():
            for trial_name, trial_data in metric_data.items():
                data = pd.DataFrame(trial_data)
                a = data.quantile(0.75)
                b = data.quantile(0.25)
                c = data
                c[
                    (c >= (a - b) * 1.5 + a) | (c <= b - (a - b) * outlier_constant)
                ] = np.nan
                c.fillna(c.median(), inplace=True)
                self.metrics[metric_name][trial_name] = MetricList(
                    [x[0] for x in c.values.tolist()]
                )

    def transpose(self):
        transposed_metrics = OrderedDict()
        for metric_name in self.metrics.keys():
            for trial_tag_list in self.metrics[metric_name].keys():
                if trial_tag_list not in transposed_metrics:
                    transposed_metrics[trial_tag_list] = {}
                transposed_metrics[trial_tag_list][metric_name] = self.metrics[
                    metric_name
                ][trial_tag_list]
        return transposed_metrics

    def _rank_test_by_trial(self, **kwargs):
        transposed_metrics = self.transpose()
        for trial in transposed_metrics.keys():
            self.trial_rank_test_result[trial] = {}
            for metric1 in transposed_metrics[trial].keys():
                for metric2 in transposed_metrics[trial].keys():
                    if metric1 != metric2:
                        result = ranksums(
                            transposed_metrics[trial][metric1],
                            transposed_metrics[trial][metric2],
                            kwargs.get("rank_type", "two-sided"),
                        )
                        self.trial_rank_test_result[trial][
                            "{}<->{}".format(metric1, metric2)
                        ] = result

        return self.trial_rank_test_result

    def rank_test_by_trail(self, trial, **kwargs):
        self._rank_test_by_trial(**kwargs)
        try:
            return self.trial_rank_test_result[trial]
        except KeyError:
            return self.metric_rank_test_result

    def _rank_test_by_metric(self, **kwargs):
        trial_tag_list = list(self.transpose().keys())
        for metric in self.metrics.keys():
            self.metric_rank_test_result[metric] = {}
            for trial1 in trial_tag_list:
                for trial2 in trial_tag_list:
                    if trial1 != trial2:
                        result = ranksums(
                            self.metrics[metric][trial1],
                            self.metrics[metric][trial2],
                            kwargs.get("rank_type", "two-sided"),
                        )
                        self.metric_rank_test_result[metric][
                            "{}<->{}".format(trial1, trial2)
                        ] = result
        return self.metric_rank_test_result

    def rank_test_by_metric(self, metric=None, **kwargs):
        self._rank_test_by_metric(**kwargs)
        try:
            return self.metric_rank_test_result[metric]
        except KeyError:
            return self.metric_rank_test_result

    def _get_table_data(self, **kwargs):
        table_data = []
        if kwargs.get("transpose", False):
            header = [
                "Trial",
                "Metric",
                "Values",
                "Average",
                "Median",
                "Std",
                "IQR",
                "Min",
                "Max",
            ]
            transposed_metrics = self.transpose()
            for i, trial in enumerate(transposed_metrics.keys()):
                trials = transposed_metrics[trial]
                for j, metric in enumerate(trials.keys()):
                    _data = []
                    _data += [
                        [trial, metric, [round(x, 2) for x in trials[metric][:10]]]
                    ]
                    _data[-1].append(round(trials[metric].avg, 2))
                    _data[-1].append(round(trials[metric].median, 2))
                    _data[-1].append(round(trials[metric].std, 2))
                    _data[-1].append(round(trials[metric].iqr, 2))
                    _data[-1].append(round(trials[metric].min, 2))
                    _data[-1].append(round(trials[metric].max, 2))
                    table_data += _data

        else:
            header = [
                "Metric",
                "Trial",
                "Values",
                "Average",
                "Median",
                "Std",
                "IQR",
                "Min",
                "Max",
            ]
            transposed_metrics = self.metrics
            for i, metric in enumerate(transposed_metrics.keys()):
                metrics = transposed_metrics[metric]
                for j, trial in enumerate(metrics.keys()):
                    _data = []
                    _data += [
                        [metric, trial, [round(x, 2) for x in metrics[trial][:10]]]
                    ]
                    _data[-1].append(round(metrics[trial].avg, 2))
                    _data[-1].append(round(metrics[trial].median, 2))
                    _data[-1].append(round(metrics[trial].std, 2))
                    _data[-1].append(round(metrics[trial].iqr, 2))
                    _data[-1].append(round(metrics[trial].min, 2))
                    _data[-1].append(round(metrics[trial].max, 2))
                    table_data += _data

        return table_data, header

    def summary(self, save_path=None, filename=None, no_print=False, **kwargs):
        if filename:
            print("Warning: filename is deprecated, please use save_path instead.")

        table_data, header = self._get_table_data(**kwargs)

        summary_str = tabulate(
            table_data, headers=header, numalign="center", tablefmt="fancy_grid"
        )
        logo = " Metric Visualizer "
        url = " https://github.com/yangheng95/metric_visualizer "
        _prefix = (
            "\n"
            + "-" * ((len(summary_str.split("\n")[0]) - len(logo)) // 2)
            + logo
            + "-" * ((len(summary_str.split("\n")[0]) - len(logo)) // 2)
            + "\n"
        )

        _postfix = (
            "\n"
            + "-" * ((len(summary_str.split("\n")[0]) - len(url)) // 2)
            + url
            + "-" * ((len(summary_str.split("\n")[0]) - len(url)) // 2)
            + "\n"
        )

        summary_str = _prefix + summary_str + _postfix
        if not no_print:
            print(summary_str)

        if save_path:
            if not save_path.endswith(".summary.txt"):
                save_path = save_path + ".summary.txt"

            fout = open(save_path, mode="w", encoding="utf8")
            summary_str += "\n{}\n".format(str(self.metrics))
            fout.write(summary_str)
            fout.close()

            self.dump(save_path.replace(".summary.txt", ".mv"))

        return summary_str

    def drop(self, *, metric=None, trial=None):
        if metric:
            self.metrics.pop(metric)
        if trial:
            for metric in self.metrics.keys():
                self.metrics[metric].pop(trial)

    def fillna(self, value=0):
        for metric in self.metrics.keys():
            for trial in self.metrics[metric].keys():
                for i, x in enumerate(self.metrics[metric][trial]):
                    if x == np.nan or x == np.inf or x == -np.inf or x is None:
                        self.metrics[metric][trial] = value

    def to_execl(self, path=None, **kwargs):
        """Save the metrics to an excel file

        :param path:  the path to save the excel file
        :param kwargs:  the kwargs to pass to the _get_table_data function
        :return:
        """
        if not path:
            path = os.getcwd() + "/{}.xlsx".format(self.name)
        if not path.endswith(".xlsx"):
            path = path + ".xlsx"

        writer = pd.ExcelWriter(path, engine="xlsxwriter")
        table_data, header = self._get_table_data(**kwargs)

        df = pd.DataFrame(table_data, columns=header)
        df.to_excel(writer, sheet_name=self.name)
        writer.save()

    def to_txt(self, path=None, **kwargs):
        """Save the metrics to a txt file

        :param path:  the path to save the txt file
        :param kwargs:  the kwargs to pass to the _get_table_data function
        :return:
        """
        if not path:
            path = os.getcwd() + "/{}.txt".format(self.name)
        if not path.endswith(".txt"):
            path = path + ".txt"

        with open(path, "w") as f:
            table_data, header = self._get_table_data(**kwargs)
            f.write(
                tabulate(
                    table_data, headers=header, numalign="center", tablefmt="fancy_grid"
                )
            )

    def to_csv(self, path=None, **kwargs):
        """Save the metrics to a csv file

        :param path:  the path to save the csv file
        :param kwargs:  the kwargs to pass to the _get_table_data function
        :return:
        """
        if not path:
            path = os.getcwd() + "/{}.csv".format(self.name)
        if not path.endswith(".csv"):
            path = path + ".csv"

        table_data, header = self._get_table_data(**kwargs)
        df = pd.DataFrame(table_data, columns=header)
        df.to_csv(path, index=kwargs.get("index", False))

    def to_json(self, path=None, **kwargs):
        """Save the metrics to a json file

        :param path:  the path to save the json file
        :param kwargs:  the kwargs to pass to the _get_table_data function
        :return:
        """
        if not path:
            path = os.getcwd() + "/{}.json".format(self.name)
        if not path.endswith(".json"):
            path = path + ".json"
        with open(path, mode="w", encoding="utf8") as fout:
            json.dump(self.metrics, fout, indent=4)

    def to_latex(self, path=None, **kwargs):
        """Save the metrics to a latex file

        :param path:  the path to save the latex file
        :param kwargs:  the kwargs to pass to the _get_table_data function
        :return:
        """
        if not path:
            path = os.getcwd() + "/{}.tex".format(self.name)
        if not path.endswith(".tex"):
            path = path + ".tex"
        with open(path, mode="w", encoding="utf8") as fout:
            table_data, header = self._get_table_data(**kwargs)
            fout.write(
                tabulate(
                    table_data, headers=header, numalign="center", tablefmt="latex"
                )
            )

    def dump(self, filename=None):
        """Dump the metric visualizer to a file

        :param filename:  the file name (or path) to dump the metric visualizer
        :return:
        """
        t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if not filename:
            filename = self.name + t
        if not filename.endswith(".mv"):
            filename = filename + ".mv"
        with open(filename, mode="wb") as fout:
            pickle.dump(self, fout)

    @staticmethod
    def load(filename=None) -> "MetricVisualizer":
        """
        Load the metric visualizer from a file

        :param filename:  the file name (or path) of the metric visualizer
        :return: A metric visualizer object
        """
        mv = None

        if not filename:
            filenames = find_cwd_files(".mv")
        elif isinstance(filename, str):
            filenames = [filename]
        elif isinstance(filename, list):
            filenames = filename
        else:
            raise ValueError("The filename should be a string or a list of strings")

        for fn in filenames:
            if not os.path.exists(fn):
                fn = find_cwd_files([fn, ".mv"])

            print("Load", fn)
            if not mv:
                mv = pickle.load(open(fn, mode="rb"))
            else:
                _ = pickle.load(open(fn, mode="rb"))
                for metric_name in _.metrics:
                    if metric_name not in mv.metrics:
                        mv.metrics[metric_name] = []
                    mv.metrics[metric_name].update(_.metrics[metric_name])
        return mv

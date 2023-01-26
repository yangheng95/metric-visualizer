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

        self.trial_rank_test_result = None
        self.metric_rank_test_result = None

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
        self.trial2unit[trial_name] = unit

        # if trial_name is None, use the length of the metric dict as the trial name
        if trial_name is None:
            trial_name = "Trial{}".format(
                len(self.metrics[metric_name]) + 1 if metric_name in self.metrics else 1
            )

        # add the metric to the metric dict
        if metric_name in self.metrics:
            if trial_name not in self.metrics[metric_name]:
                self.metrics[metric_name][trial_name] = [value]
            else:
                self.metrics[metric_name][trial_name].append(value)
        else:
            self.metrics[metric_name] = {trial_name: [value]}

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

        if not kwargs.get("markers", None):
            markers = self.MARKERS[:]
        else:
            markers = kwargs.pop("markers")

        if not kwargs.get("colors", None):
            colors = self.COLORS[:]
        else:
            colors = kwargs.pop("colors")

        box_parts = []
        ax = plt.subplot()
        for metric_name in plot_metrics.keys():
            metrics = plot_metrics[metric_name]

            color = random.choice(colors)
            colors.remove(color)

            data = [metrics[trial] for trial in metrics.keys()]

            boxs_parts = ax.boxplot(
                data,
                positions=list(range(len(metrics.keys()))),
                widths=kwargs.pop("widths", 0.8),
                meanline=True,
                boxprops={'linewidth': '3'},
                medianprops={'linewidth': '3', 'color': 'red'},
                meanprops={'linewidth': '3'},
                capprops={'linewidth': '3'},
                flierprops={"marker": "o", "markerfacecolor": "k", "markersize": 10},
                whiskerprops={'linewidth': '3'}
            )

            box_parts.append(boxs_parts["boxes"][0])

            for item in ["boxes", "whiskers", "fliers", "medians", "caps"]:
                plt.setp(boxs_parts[item], color=color)

            plt.setp(boxs_parts["fliers"], markeredgecolor=color)

        if kwargs.get("legend", True):
            ax.legend(box_parts, list(plot_metrics.keys()), loc=kwargs.pop("legend_loc", 1))

        if kwargs.get("minor_ticks", True):
            ax.minorticks_on()

        if kwargs.get("grid", True):
            ax.grid(which="major", linestyle="-", linewidth="0.3", color="grey")
            ax.grid(which="minor", linestyle=":", linewidth="0.3", color="grey")

        plt.xticks(
            kwargs.get("xticks", list(range(len(metrics.keys())))),
            labels=list(metrics.keys()),
            rotation=kwargs.pop("xrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            # verticalalignment=kwargs.pop("verticalalignment", "top"),
            **kwargs.pop("xticks_kwargs", {}),
        )
        plt.yticks(
            rotation=kwargs.pop("yrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            verticalalignment=kwargs.pop("verticalalignment", "center_baseline"),
            **kwargs.pop("yticks_kwargs", {}),
        )

        plt.xlabel(kwargs.pop("xlabel", "Trial Name"))
        plt.ylabel(kwargs.pop("ylabel", "Metric Value"))

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
                plt.savefig(save_path)
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

        if not kwargs.get("markers", None):
            markers = self.MARKERS[:]
        else:
            markers = kwargs.pop("markers")

        if not kwargs.get("colors", None):
            colors = self.COLORS[:]
        else:
            colors = kwargs.pop("colors")

        violin_parts = []
        ax = plt.subplot()
        for metric_name in plot_metrics.keys():
            metrics = plot_metrics[metric_name]
            data = [metrics[trial] for trial in metrics.keys()]

            violin = ax.violinplot(
                data,
                widths=kwargs.pop("widths", 0.8),
                positions=list(range(len(metrics.keys()))),
                showmeans=True,
                showmedians=True,
                showextrema=True,
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
            kwargs.get("xticks", list(range(len(metrics.keys())))),
            list(metrics.keys()),
            rotation=kwargs.pop("xrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            # verticalalignment=kwargs.pop("verticalalignment", "top"),
            **kwargs.pop("xticks_kwargs", {}),
        )
        plt.yticks(
            rotation=kwargs.pop("yrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            verticalalignment=kwargs.pop("verticalalignment", "center_baseline"),
            **kwargs.pop("yticks_kwargs", {}),
        )

        plt.xlabel(kwargs.pop("xlabel", "Trial Name"))
        plt.ylabel(kwargs.pop("ylabel", "Metric Value"))

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
                plt.savefig(save_path)
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

        scatter_parts = []
        ax = plt.subplot()
        # draw the scatter plot
        for i, metric_name in enumerate(plot_metrics.keys()):
            # get the values
            metrics = plot_metrics[metric_name]
            values = list(plot_metrics[metric_name].values())
            # draw the scatter plot
            scatter_part = ax.scatter(
                list(range(len(values))),
                values,
                color=colors[i],
                **kwargs.get("scatter_kwargs", {}),
            )

            scatter_parts.append(scatter_part["bodies"][0])

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
            kwargs.get("xticks", list(range(len(metrics.keys())))),
            list(metrics.keys()),
            rotation=kwargs.pop("xrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            # verticalalignment=kwargs.pop("verticalalignment", "top"),
            **kwargs.pop("xticks_kwargs", {}),
        )
        plt.yticks(
            rotation=kwargs.pop("yrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            verticalalignment=kwargs.pop("verticalalignment", "center_baseline"),
            **kwargs.pop("yticks_kwargs", {}),
        )

        plt.xlabel(kwargs.pop("xlabel", "Trial Name"))
        plt.ylabel(kwargs.pop("ylabel", "Metric Value"))

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
                plt.savefig(save_path)
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

            # y_avg = np.median(y, axis=1)

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
            list(metrics.keys()),
            rotation=kwargs.pop("xrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            # verticalalignment=kwargs.pop("verticalalignment", "top"),
            **kwargs.pop("xticks_kwargs", {}),
        )
        plt.yticks(
            rotation=kwargs.pop("yrotation", 0),
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            verticalalignment=kwargs.pop("verticalalignment", "center_baseline"),
            **kwargs.pop("yticks_kwargs", {}),
        )

        plt.xlabel(kwargs.pop("xlabel", "Trial Name"))
        plt.ylabel(kwargs.pop("ylabel", "Metric Value"))

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
                plt.savefig(save_path)
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
            plt.legend(bar_parts, list(plot_metrics.keys()), loc=kwargs.pop("legend_loc", 1))

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
            horizontalalignment=kwargs.pop("horizontalalignment", "center"),
            verticalalignment=kwargs.pop("verticalalignment", "center_baseline"),
            **kwargs.pop("yticks_kwargs", {}),
        )

        plt.xlabel(kwargs.pop("xlabel", "Trial Name"))
        plt.ylabel(kwargs.pop("ylabel", "Metric Value"))

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
                plt.savefig(save_path)
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
                                 + list(plot_metrics.keys())[target_trial + 1:]
                },
                "medium": {
                    trial: [0]
                    for trial in list(plot_metrics.keys())[:target_trial]
                                 + list(plot_metrics.keys())[target_trial + 1:]
                },
                "small": {
                    trial: [0]
                    for trial in list(plot_metrics.keys())[:target_trial]
                                 + list(plot_metrics.keys())[target_trial + 1:]
                },
                "equal": {
                    trial: [0]
                    for trial in list(plot_metrics.keys())[:target_trial]
                                 + list(plot_metrics.keys())[target_trial + 1:]
                },
            }
            max_num = 0
            count = 0
            trial1 = list(plot_metrics.keys())[target_trial]
            for trial2 in (
                    list(plot_metrics.keys())[:target_trial]
                    + list(plot_metrics.keys())[target_trial + 1:]
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
                self.metrics[metric_name][trial_name] = [
                    x[0] for x in c.values.tolist()
                ]

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

        table_data = []
        if kwargs.get("transpose", False):
            transposed_metrics = self.transpose()
        else:
            transposed_metrics = self.metrics
        for mn in transposed_metrics.keys():
            metrics = transposed_metrics[mn]
            trial_tag_list = list(self.trial2unit.keys())
            for i, trial in enumerate(metrics.keys()):
                _data = []
                _data += [
                    [mn, trial_tag_list[i], [round(x, 2) for x in metrics[trial][:10]]]
                ]
                _data[-1].append(round(np.mean(metrics[trial]), 2))
                _data[-1].append(round(np.median(metrics[trial]), 2))
                _data[-1].append(round(np.std(metrics[trial]), 2))
                _data[-1].append(
                    round(
                        np.percentile(metrics[trial], 75)
                        - np.percentile(metrics[trial], 25),
                        2,
                    )
                )
                _data[-1].append(round(np.min(metrics[trial]), 2))
                _data[-1].append(round(np.max(metrics[trial]), 2))

                table_data += _data

        return table_data, header

    def summary(self, dump_path=os.getcwd(), filename=None, no_print=False, **kwargs):
        summary_str = "\n ----------------------------------- Metric Visualizer ----------------------------------- \n"

        table_data, header = self._get_table_data(**kwargs)

        summary_str += tabulate(
            table_data, headers=header, numalign="center", tablefmt="fancy_grid"
        )
        summary_str += "\n -------------------- https://github.com/yangheng95/metric_visualizer --------------------\n"
        summary_str += "\n -------------- You can use: mvis *.mv to visualize the metrics in any bash --------------\n"
        if not no_print:
            print(summary_str)

        if dump_path:
            prefix = os.path.join(dump_path, self.name if self.name else "")
            if filename:
                prefix = prefix + filename
            fout = open(prefix + ".summary.txt", mode="w", encoding="utf8")
            summary_str += "\n{}\n".format(str(self.metrics))
            fout.write(summary_str)
            fout.close()

            self.dump()

        return summary_str

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
    def load(filename=None):
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

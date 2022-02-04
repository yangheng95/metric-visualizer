# -*- coding: utf-8 -*-
# file: metric_visualizer.py
# time: 03/02/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os.path

import matplotlib.colors
import numpy as np
import tikzplotlib
from findfile import find_cwd_files
from matplotlib import pyplot as plt
from scipy.stats import iqr


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


class MetricVisualizer:
    COLORS_DICT = matplotlib.colors.XKCD_COLORS
    COLORS_DICT.update(matplotlib.colors.CSS4_COLORS)
    COLORS = list(COLORS_DICT.values())
    # MARKERS = matplotlib.markers
    MARKERS = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_",
               0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, "None", " ", ""]
    HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    box_plot_tex_template = r"""
    \documentclass{article}
    \usepackage{pgfplots}
    \usepackage{tikz}
    \usepackage{caption}
    \usetikzlibrary{intersections}
    \usepackage{helvet}
    \usepackage[eulergreek]{sansmath}
    \usepackage{amsfonts,amssymb,amsmath,amsthm,amsopn}	% math related

    \begin{document}
        \pagestyle{empty}
        \pgfplotsset{ compat=1.12,every axis/.append style={
            grid = major,
            thick,
            xtick={$xtick$},
            xticklabels={$xticklabels$},
            ylabel = {$ylabel$},
            ylabel style={font=\Large},
            xlabel = {$xlabel$},
            xlabel style={font=\Large},
            x tick label style={rotate=20,anchor=north},
            y tick label style={rotate=90,anchor=south},
            xticklabel shift=1pt,
            line width = 1pt,
            tick style = {line width = 0.8pt}}}
        \pgfplotsset{every plot/.append style={thin}}


        \begin{figure}
        \centering

        $tikz_code$

        \end{figure}

    \end{document}
    """

    violin_plot_tex_template = r"""
    \documentclass{article}
    \usepackage{pgfplots}
    \usepackage{tikz}
    \usepackage{caption}
    \usetikzlibrary{intersections}
    \usepackage{helvet}
    \usepackage[eulergreek]{sansmath}
    \usepackage{amsfonts,amssymb,amsmath,amsthm,amsopn}	% math related

    \begin{document}
        \pagestyle{empty}
        \pgfplotsset{ compat=1.12,every axis/.append style={
            grid = major,
            thick,
            xtick={$xtick$},
            xticklabels={$xticklabels$},
            ylabel = {$ylabel$},
            ylabel style={font=\Large},
            xlabel = {$xlabel$},
            xlabel style={font=\Large},
            x tick label style={rotate=20,anchor=north},
            y tick label style={rotate=90,anchor=south},
            xticklabel shift=1pt,
            line width = 1pt,
            tick style = {line width = 0.8pt}}}
        \pgfplotsset{every plot/.append style={thin}}


        \begin{figure}
        \centering

        $tikz_code$

        \end{figure}

    \end{document}
    """

    traj_plot_tex_template = r"""
    \documentclass{article}
    \usepackage{pgfplots}
    \usepackage{tikz}
    \usetikzlibrary{intersections}
    \usepackage{helvet}
    \usepackage[eulergreek]{sansmath}

    \begin{document}
        \pagestyle{empty}
        \pgfplotsset{every axis/.append style={
            grid = major,
            thick,
            xtick={$xtick$},
            xticklabels={$xticklabels$},
            xlabel = {$xlabel$},
            xlabel style={font=\Large},
            x tick label style={rotate=20,anchor=north},
            y tick label style={rotate=90,anchor=south},
            ylabel = {$ylabel$},
            ylabel style={font=\Large},
            xticklabel shift=1pt,
            line width = 1pt,
            tick style = {line width = 0.8pt}}}
        }
        \pgfplotsset{every plot/.append style={very thin}}


        \begin{figure}
        \centering

        $tikz_code$

        \end{figure}

    \end{document}
"""

    def set_box_plot_tex_template(self, box_plot_tex_template):
        self.box_plot_tex_template = box_plot_tex_template

    def set_violin_plot_tex_template(self, box_violin_tex_template):
        self.box_plot_tex_template = box_violin_tex_template

    def set_traj_plot_tex_template(self, box_traj_tex_template):
        self.box_plot_tex_template = box_traj_tex_template

    def __init__(self, metric_dict=None):
        if metric_dict is None:
            self.metrics = {
                # 'Metric1': [80.41, 79.78, 81.03, 80.09, 79.62, 80.56, 80.88, 79.94, 79.47, 79.78, 80.72, 79.78, 81.35, 80.88, 81.03],
                # 'Metric2': [76.79, 75.49, 77.92, 77.21, 75.63, 76.96, 77.44, 76.26, 76.35, 76.12, 76.12, 76.78, 75.64, 77.31, 73.79]
            }

    def clear(self):
        self.metrics = {}

    def add_metric(self, metric_name='Accuracy', value=0):
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            self.metrics[metric_name] = [value]

    def traj_plot(self, save_path=None, **kwargs):

        ax = plt.subplot()

        alpha = kwargs.pop('markersize', 0.5)

        markersize = kwargs.pop('markersize', 3)

        markers = kwargs.pop('markers', self.MARKERS)

        plot_labels = kwargs.pop('plot_labels', None)

        xticks = kwargs.pop('xticks', '')

        xlabel = kwargs.pop('xlabel', '')

        ylabel = kwargs.pop('ylabel', '')

        colors = kwargs.pop('colors', self.COLORS)

        hatches = kwargs.pop('hatches', None)

        fontsize = kwargs.pop('fontsize', 12)

        rotation = kwargs.pop('rotation', 0)

        x = [i for i in range(len(self.metrics))] if not xticks else xticks
        y = np.array([self.metrics[metric_name] for metric_name in self.metrics])

        y_avg = np.average(y, axis=1)
        y_std = np.std(y, axis=1)

        for i in range(len(x)):
            if kwargs.pop('avg_point', True):
                ax.plot(x,
                        y_avg,
                        marker=markers[i],
                        color=colors[i],
                        label=plot_labels[i] if plot_labels else None,
                        markersize=markersize,
                        )
            if kwargs.pop('traj_point', True):
                ax.plot(x,
                        y,
                        marker=markers[i],
                        color=colors[i],
                        # label=plot_labels[i],
                        markersize=markersize,
                        )
            if kwargs.pop('traj_fill', True):
                plt.fill_between(x,
                                 y_avg - y_std,
                                 y_avg + y_std,
                                 color=colors[i],
                                 hatch=hatches[i] if hatches else None,
                                 alpha=alpha
                                 )

        plt.grid()
        plt.minorticks_on()
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize, rotation=rotation)

        ax.grid()
        ax.minorticks_on()

        legend_without_duplicate_labels(ax)

        tikz_code = tikzplotlib.get_tikz_code()
        tex_src = self.violin_plot_tex_template.replace('$tikz_code$', tikz_code)

        tex_src = tex_src.replace('$xtick$', xticks)
        tex_src = tex_src.replace('$xticklabels$', ', '.join(list(range(len(xticks)))))
        tex_src = tex_src.replace('$xlabel$', xlabel)
        tex_src = tex_src.replace('$ylabel$', ylabel)

        if not save_path:
            plt.show()
        else:
            # plt.savefig(save_path, dpi=1000, format='pdf')
            open(save_path + '_metric_traj_plot.tex', mode='w', encoding='utf8').write(tex_src)
            texs = find_cwd_files('.tex')
            for pdf in texs:
                cmd = 'pdflatex "{}"'.format(pdf).replace(os.path.sep, '/')
                os.system(cmd)

            pdfs = find_cwd_files('.pdf', exclude_key='crop')
            for pdf in pdfs:
                cmd = 'pdfcrop "{}" "{}"'.format(pdf, pdf).replace(os.path.sep, '/')
                os.system(cmd)

            for f in find_cwd_files(['.aux']) + find_cwd_files(['.log']) + find_cwd_files(['crop']):
                os.remove(f)

        plt.close()

    def box_plot(self, save_path=None, **kwargs):
        ax = plt.subplot()

        ax = plt.subplot()

        alpha = kwargs.pop('markersize', 0.5)

        markersize = kwargs.pop('markersize', 3)

        markers = kwargs.pop('markers', self.MARKERS)

        plot_labels = kwargs.pop('plot_labels', None)

        xticks = kwargs.pop('xticks', '')

        xlabel = kwargs.pop('xlabel', '')

        ylabel = kwargs.pop('ylabel', '')

        colors = kwargs.pop('colors', self.COLORS)

        hatches = kwargs.pop('hatches', None)

        fontsize = kwargs.pop('fontsize', 12)

        rotation = kwargs.pop('rotation', 0)

        data = [self.metrics[metric_name] for metric_name in self.metrics]

        widths = kwargs.pop('widths', 0.5)

        boxs_parts = ax.boxplot(data, widths=widths)

        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(boxs_parts[item], color='grey')

        plt.setp(boxs_parts["fliers"], markeredgecolor='grey')

        legend_without_duplicate_labels(ax)

        tikz_code = tikzplotlib.get_tikz_code()
        tex_src = self.box_plot_tex_template.replace('$tikz_code$', tikz_code)

        tex_src = tex_src.replace('$xtick$', xticks)
        tex_src = tex_src.replace('$xticklabels$', ', '.join(list(range(len(xticks)))))
        tex_src = tex_src.replace('$xlabel$', xlabel)
        tex_src = tex_src.replace('$ylabel$', ylabel)

        if not save_path:
            plt.show()
        else:
            # plt.savefig(save_path, dpi=1000, format='pdf')
            open(save_path + '_metric_box_plot.tex', mode='w', encoding='utf8').write(tex_src)
            texs = find_cwd_files('.tex')
            for pdf in texs:
                cmd = 'pdflatex "{}"'.format(pdf).replace(os.path.sep, '/')
                os.system(cmd)

            pdfs = find_cwd_files('.pdf', exclude_key='crop')
            for pdf in pdfs:
                cmd = 'pdfcrop "{}" "{}"'.format(pdf, pdf).replace(os.path.sep, '/')
                os.system(cmd)

            for f in find_cwd_files(['.aux']) + find_cwd_files(['.log']) + find_cwd_files(['crop']):
                os.remove(f)

        plt.close()

    def violin_plot(self, save_path=None, **kwargs):
        ax = plt.subplot()

        alpha = kwargs.pop('markersize', 0.5)

        markersize = kwargs.pop('markersize', 3)

        markers = kwargs.pop('markers', self.MARKERS)

        plot_labels = kwargs.pop('plot_labels', None)

        xticks = kwargs.pop('xticks', '')

        xlabel = kwargs.pop('xlabel', '')

        ylabel = kwargs.pop('ylabel', '')

        colors = kwargs.pop('colors', self.COLORS)

        hatches = kwargs.pop('hatches', None)

        fontsize = kwargs.pop('fontsize', 12)

        rotation = kwargs.pop('rotation', 0)

        linewidth = kwargs.pop('linewidth', 2)

        data = [self.metrics[metric_name] for metric_name in self.metrics]

        violin_parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)

        for pc in violin_parts['bodies']:
            # pc.set_facecolor('black')
            # pc.set_edgecolor('black')
            pc.set_linewidth(linewidth)

        legend_without_duplicate_labels(ax)

        tikz_code = tikzplotlib.get_tikz_code()
        tex_src = self.box_plot_tex_template.replace('$tikz_code$', tikz_code)

        tex_src = tex_src.replace('$xtick$', xticks)
        tex_src = tex_src.replace('$xticklabels$', ', '.join(list(range(len(xticks)))))
        tex_src = tex_src.replace('$xlabel$', xlabel)
        tex_src = tex_src.replace('$ylabel$', ylabel)

        if not save_path:
            plt.show()
        else:
            # plt.savefig(save_path, dpi=1000, format='pdf')
            open(save_path + '_metric_box_plot.tex', mode='w', encoding='utf8').write(tex_src)
            texs = find_cwd_files('.tex')
            for pdf in texs:
                cmd = 'pdflatex "{}"'.format(pdf).replace(os.path.sep, '/')
                os.system(cmd)

            pdfs = find_cwd_files('.pdf', exclude_key='crop')
            for pdf in pdfs:
                cmd = 'pdfcrop "{}" "{}"'.format(pdf, pdf).replace(os.path.sep, '/')
                os.system(cmd)

            for f in find_cwd_files(['.aux']) + find_cwd_files(['.log']) + find_cwd_files(['crop']):
                os.remove(f)
        plt.close()

    def summary(self, save_path=None, **kwargs):
        summary_str = ''
        summary_str += ' -------------------- Metric Summary --------------------\n'
        for metric_name in self.metrics:
            metrics = self.metrics[metric_name]
            summary_str += '{}: '.format(metric_name) + str(metrics) + '\n'
            summary_str += 'Avg:{}, \tMedian: {}, \tIQR: {}, \tMax: {}, \tMin: {}\n'.format(
                np.average(metrics),
                np.median(metrics),
                iqr(metrics, rng=(25, 75), interpolation='midpoint'),
                np.max(metrics),
                np.min(metrics)
            )
        summary_str += ' -------------------- Metric Summary --------------------\n'

        print(summary_str)

        if save_path:
            fout = open(save_path + '.txt', mode='w', encoding='utf8')
            fout.write(summary_str)
            fout.close()


if __name__ == '__main__':
    mv = MetricVisualizer()
    mv.summary()
    mv.traj_plot(save_plot=True)
    mv.violin_plot()
    mv.box_plot()

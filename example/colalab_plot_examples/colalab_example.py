# -*- coding: utf-8 -*-
# file: colalab_example.py
# time: 19:29 2023/1/10
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
from metric_visualizer.colalab import reformat_tikz_format_for_colalab
from metric_visualizer import MetricVisualizer

if __name__ == "__main__":

    import os
    import random
    from metric_visualizer import MetricVisualizer
    import numpy as np

    MV = MetricVisualizer(name="example", trial_tag="Model")

    repeat = 100  # number of repeats
    metric_num = 3  # number of metrics

    #  利用metric_visualizer监听实验吧并保存实验结果，随时重新绘制图像
    # trial_names = ["LSTM", "CNN", "BERT"]  # fake trial names
    trial_names = ["NSGA-II", "NSGA-III", "MOEA/D"]  # fake trial names
    # trial_names = ['Hyperparameter Setting 1', 'Hyperparameter Setting 2', 'Hyperparameter Setting 3']  # fake trial names

    for n_trial in range(len(trial_names)):
        for r in range(repeat):  # repeat the experiments to plot violin or box figure
            metrics = [
                (np.random.random() + n + (1 if random.random() > 0.5 else -1))
                for n in range(metric_num)
            ]  # n is metric scale factor
            for i, m in enumerate(metrics):
                MV.add_metric(
                    metric_name="metric{}".format(i + 1), value=m
                )  # add metric by custom name and value
        MV.next_trial()  # next_trial() should be used with add_metric() to add metrics of different trials

    MV.remove_outliers()  # remove outliers

    # MV.summary(dump_path=os.getcwd(), filename="file_name", no_print=True)
    # MV.traj_plot_by_trial(xlabel="", xrotation=30, minorticks_on=True)
    # MV.violin_plot_by_trial()
    # MV.box_plot_by_trial()
    # MV.avg_bar_plot_by_trial()
    # MV.sum_bar_plot_by_trial()
    #
    # MV.traj_plot_by_metric(xlabel="", xrotation=30, minorticks_on=True)
    # MV.violin_plot_by_metric()
    # MV.box_plot_by_metric()
    # MV.avg_bar_plot_by_metric()
    # MV.sum_bar_plot_by_metric()
    #
    # MV.scott_knott_plot(plot_type="box", minorticks_on=False)
    # MV.scott_knott_plot(
    #     plot_type="violin", minorticks_on=False
    # )  # save fig_preview into .texg and .pdf format
    #
    # # MV.A12_bar_plot()  # need to install R language and rpy2 package
    #
    # rank_test_result = MV.rank_test_by_trail("trial1")
    # rank_test_result = MV.rank_test_by_metric("metric1")
    #
    # print(MV.rank_test_by_trail("trial0"))
    # print(MV.rank_test_by_metric("metric1"))

    # select a tikz template, paste the path of the template or text of the template
    tex_src_template = r"""
        \documentclass{article}
        \usepackage{pgfplots}
        \usepackage{tikz}
        \usetikzlibrary{intersections}
        \usepackage{helvet}
        \usepackage[eulergreek]{sansmath}

        \begin{document}
        \pagestyle{empty}

        \pgfplotsset{every axis/.append style={
        font = \Large,
        grid = major,
        xlabel = {index},
        ylabel = {R-HV},
        %ymode  = log,
        thick,
        %xmin = 1,
        %xmax = 10,
        %ymin = 3.6,
        %ymax = 4.2,
        line width = 1pt,
        tick style = {line width = 0.8pt}}}

        \begin{tikzpicture}
            \begin{axis}[
                xtick = {1,...,10},
                xticklabels = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1},
                legend pos = south east,
                legend cell align=left,
                legend style={font=\footnotesize},
                legend entries = {r-MOEA/D-STM, R-NSGA-II, g-NSGA-II, r-NSGA-II},
            ]
            \addplot[black, mark = *, mark size = 2.5pt] table[x index = 0, y index = 1] {data/delta_hv.dat};
            \addplot[green, mark = square*, mark size = 2.5pt] table[x index = 0, y index = 2] {data/delta_hv.dat};
            \addplot[red, mark = triangle*, mark size = 2.5pt] table[x index = 0, y index = 3] {data/delta_hv.dat}; 
            \end{axis}
        \end{tikzpicture}

        \end{document}
            """
    MV = MV.load()
    MV.to_execl()
    tex_src_data = MV.box_plot_by_metric()

    style_settings = {
        "legend pos": "north west",
        "legend entries": "{}",
        "xlabel": "This is X label",
        "ylabel": "This is Y label",
        "xtick": "{0,1,2}",
        "xticklabels": "{0,1,2}",
        # write your own style settings here
        # it will be appended to the tikz picture style settings
    }
    reformat_tikz_format_for_colalab(
        tex_src_template,
        tex_src_data,
        output_path="colalab_example.tex",
        style_settings=style_settings,
    )

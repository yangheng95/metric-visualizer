# -*- coding: utf-8 -*-
# file: example.py
# time: 05/02/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import random

from metric_visualizer import MetricVisualizer
import numpy as np

MV = MetricVisualizer(name='example', trial_tag='Trial ID')

trial_num = 10  # number of different trials,
repeat = 20  # number of repeats
metric_num = 3  # number of metrics

#  利用metric_visualizer监听实验吧并保存实验结果，随时重新绘制图像

for n_trial in range(trial_num):
    for r in range(repeat):  # repeat the experiments to plot violin or box figure
        metrics = [(np.random.random() + n + (1 if random.random() > 0.5 else -1)) for n in range(metric_num)]  # n is metric scale factor
        for i, m in enumerate(metrics):
            MV.add_metric('metric{}'.format(i + 1), m)  # add metric by custom name and value
    MV.next_trial()

MV.summary(no_print=True)  # save fig_preview into .tex and .pdf format
MV.traj_plot_by_trial(xlabel='', xrotation=30, minorticks_on=True)  # save fig_preview into .tex and .pdf format
MV.violin_plot_by_trial()  # save fig_preview into .tex and .pdf format
MV.box_plot_by_trial()  # save fig_preview into .tex and .pdf format
MV.box_plot_by_trial()  # save fig_preview into .tex and .pdf format
MV.avg_bar_plot_by_trial()  # save fig_preview into .tex and .pdf format
MV.sum_bar_plot_by_trial()  # save fig_preview into .tex and .pdf format

MV.traj_plot_by_metric(xlabel='', xrotation=30, minorticks_on=True)  # save fig_preview into .tex and .pdf format
MV.violin_plot_by_metric()  # save fig_preview into .tex and .pdf format
MV.box_plot_by_metric()  # save fig_preview into .tex and .pdf format
MV.box_plot_by_metric()  # save fig_preview into .tex and .pdf format
MV.avg_bar_plot_by_metric()  # save fig_preview into .tex and .pdf format
MV.sum_bar_plot_by_metric()  # save fig_preview into .tex and .pdf format
#
MV.scott_knott_plot(plot_type='box', minorticks_on=False)  # save fig_preview into .tex and .pdf format
MV.scott_knott_plot(plot_type='violin', minorticks_on=False)  # save fig_preview into .tex and .pdf format

MV.A12_bar_plot()  # save fig_preview into .tex and .pdf format

print(MV.rank_test_by_trail('trial0'))  # save fig_preview into .tex and .pdf format
print(MV.rank_test_by_metric('metric1'))  # save fig_preview into .tex and .pdf format

# -*- coding: utf-8 -*-
# file: example.py
# time: 05/02/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import random

from metric_visualizer import MetricVisualizer
import numpy as np

MV = MetricVisualizer(name='example', trial_tag='Trial ID', trial_tag_list=[0, 1, 2, 3, 4])

trial_num = 5  # number of different trials,
repeat = 20  # number of repeats
metric_num = 5  # number of metrics

for n_trial in range(trial_num):
    for r in range(repeat):  # repeat the experiments to plot violin or box figure
        metrics = [(np.random.random() + n + (1 if random.random() > 0.5 else -1)) for n in range(metric_num)]  # n is metric scale factor
        for i, m in enumerate(metrics):
            MV.add_metric('metric{}'.format(i + 1), m)
    MV.next_trial()

save_prefix = './'
MV.summary(save_path=save_prefix, no_print=True)  # save fig into .tex and .pdf format
MV.traj_plot_by_trial(save_path=save_prefix, xlabel='', xrotation=30, minorticks_on=True)  # save fig into .tex and .pdf format
MV.violin_plot_by_trial(save_path=save_prefix)  # save fig into .tex and .pdf format
MV.box_plot_by_trial(save_path=save_prefix)  # save fig into .tex and .pdf format
MV.avg_bar_plot_by_trial(save_path=save_prefix)  # save fig into .tex and .pdf format
MV.sum_bar_plot_by_trial(save_path=save_prefix)  # save fig into .tex and .pdf format
MV.scott_knott_plot(save_path=save_prefix, minorticks_on=False)  # save fig into .tex and .pdf format

print(MV.rank_test_by_trail('trial0'))  # save fig into .tex and .pdf format
print(MV.rank_test_by_metric('metric1'))  # save fig into .tex and .pdf format

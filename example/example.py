# -*- coding: utf-8 -*-
# file: example.py
# time: 05/02/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from metric_visualizer import MetricVisualizer
import numpy as np

MV = MetricVisualizer()

trail_num = 10  # number of different trails
repeat = 10  # number of repeats
metric_num = 3  # number of metrics

for trail in range(trail_num):
    for r in range(repeat):
        t = 0  # metric scale factor        # repeat the experiments to plot violin or box figure
        metrics = [(np.random.random() + n) * 100 for n in range(metric_num)]
        for i, m in enumerate(metrics):
            MV.add_metric('Metric-{}'.format(i + 1), round(m, 2))
    MV.next_trail()

save_path = None
MV.summary(save_path=save_path)  # save fig into .tex and .pdf format
MV.traj_plot(save_path=save_path)  # save fig into .tex and .pdf format
MV.violin_plot(save_path=save_path)  # save fig into .tex and .pdf format
MV.box_plot(save_path=save_path)  # save fig into .tex and .pdf format
MV.avg_bar_plot(save_path=save_path)  # save fig into .tex and .pdf format
MV.sum_bar_plot(save_path=save_path)  # save fig into .tex and .pdf format

save_path = 'example'
MV.traj_plot(save_path=save_path)  # show the fig via matplotlib
MV.violin_plot(save_path=save_path)  # show the fig via matplotlib
MV.box_plot(save_path=save_path)  # show the fig via matplotlib
MV.avg_bar_plot(save_path=save_path)  # save fig into .tex and .pdf format
MV.sum_bar_plot(save_path=save_path)  # save fig into .tex and .pdf format

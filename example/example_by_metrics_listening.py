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

MV = MetricVisualizer(name='example', trial_tag='Model')

repeat = 100  # number of repeats
metric_num = 3  # number of metrics

#  利用metric_visualizer监听实验吧并保存实验结果，随时重新绘制图像
trial_names = ['LSTM', 'CNN', 'BERT']  # fake trial names
# trial_names = ['NSGA-II', 'NSGA-III', 'MOEA/D']  # fake trial names
# trial_names = ['Hyperparameter Setting 1', 'Hyperparameter Setting 2', 'Hyperparameter Setting 3']  # fake trial names

for n_trial in range(len(trial_names)):
    for r in range(repeat):  # repeat the experiments to plot violin or box figure
        metrics = [(np.random.random() + n + (1 if random.random() > 0.5 else -1)) for n in range(metric_num)]  # n is metric scale factor
        for i, m in enumerate(metrics):
            # MV.add_metric(metric_name='metric{}'.format(i + 1), value=m)  # add metric by custom name and value
            MV.log_metric(trial_name=trial_names[n_trial], metric_name='metric{}'.format(i + 1), value=m)  # add metric by custom name and value
    # MV.next_trial()  # next_trial() should be used with add_metric() to add metrics of different trials

# MV.remove_outliers()  # remove outliers

MV.summary(dump_path=os.getcwd(), filename='file_name', no_print=True)
MV.traj_plot_by_trial(xlabel='', xrotation=30, minorticks_on=True)
MV.violin_plot_by_trial()
MV.box_plot_by_trial()
MV.box_plot_by_trial()
MV.avg_bar_plot_by_trial()
MV.sum_bar_plot_by_trial()

MV.traj_plot_by_metric(xlabel='', xrotation=30, minorticks_on=True)
MV.violin_plot_by_metric()
MV.box_plot_by_metric()
MV.box_plot_by_metric()
MV.avg_bar_plot_by_metric()
MV.sum_bar_plot_by_metric()

MV.scott_knott_plot(plot_type='box', minorticks_on=False)
MV.scott_knott_plot(plot_type='violin', minorticks_on=False)  # save fig_preview into .texg and .pdf format

# MV.A12_bar_plot()  # need to install R language and rpy2 package

rank_test_result = MV.rank_test_by_trail('trial1')
rank_test_result = MV.rank_test_by_metric('metric1')

print(MV.rank_test_by_trail('trial0'))
print(MV.rank_test_by_metric('metric1'))

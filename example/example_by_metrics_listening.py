# -*- coding: utf-8 -*-
# file: example.py
# time: 05/02/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import random
from metric_visualizer.core import MetricVisualizer
import numpy as np

MV = MetricVisualizer(name="example", trial_tag="Model")

repeat = 100  # number of repeats
metric_num = 3  # number of metrics

#  利用metric_visualizer监听实验吧并保存实验结果，随时重新绘制图像
trial_names = ["LSTM", "CNN", "BERT"]  # fake trial names
# trial_names = ['NSGA-II', 'NSGA-III', 'MOEA/D']  # fake trial names
# trial_names = ['Hyperparameter Setting 1', 'Hyperparameter Setting 2', 'Hyperparameter Setting 3']  # fake trial names
for r in range(repeat):  # repeat the experiments to plot violin or box figure
    metrics = [
        (np.random.random() + n + (1 if random.random() > 0.5 else -1))
        for n in range(metric_num)
    ]  # n is metric scale factor
    for n_trial in range(len(trial_names)):

        for i, m in enumerate(metrics):
            MV.log_metric(
                trial_name="Trial-{}".format(n_trial),
                metric_name="metric{}".format(i + 1),
                value=m + random.random(),
            )  # add metric by custom name and value

MV.remove_outliers()  # remove outliers

# MV.summary(dump_path=os.getcwd())
# MV.box_plot(by="trial", show=True)
# MV.box_plot(by="metric", show=True)
# MV.violin_plot(by="trial", show=True)
# MV.violin_plot(by="metric", show=True)
# MV.bar_plot(by="trial", show=True)
# MV.bar_plot(by="metric", show=True)
# MV.bar_plot(by="trial", show=True)
# MV.bar_plot(by="metric", show=True)
# MV.trajectory_plot(by="trial", show=True)
# MV.trajectory_plot(by="metric", show=True)
# MV.sk_rank_plot(plot_type="box", show=True)
# MV.sk_rank_plot(plot_type="violin", show=True)
# MV.a12_bar_plot(show=True)


MV.summary(dump_path=os.getcwd())
MV.box_plot(by="trial", engine="tikz", show=True, save_path=os.getcwd() + '/box_plot_by_trial.tex')
MV.box_plot(by="metric", engine="tikz", show=True, save_path=os.getcwd() + '/box_plot_by_metric.tex')
MV.violin_plot(by="trial", engine="tikz", show=True, save_path=os.getcwd() + '/violin_plot_by_trial.tex')
MV.violin_plot(by="metric", engine="tikz", show=True, save_path=os.getcwd() + '/violin_plot_by_metric.tex')
MV.bar_plot(by="trial", engine="tikz", show=True, save_path=os.getcwd() + '/bar_plot_by_trial.tex')
MV.bar_plot(by="metric", engine="tikz", show=True, save_path=os.getcwd() + '/bar_plot_by_metric.tex')
MV.bar_plot(by="trial", engine="tikz", show=True, save_path=os.getcwd() + '/bar_plot_by_trial.tex')
MV.bar_plot(by="metric", engine="tikz", show=True, save_path=os.getcwd() + '/bar_plot_by_metric.tex')
MV.trajectory_plot(by="trial", engine="tikz", show=True, save_path=os.getcwd() + '/trajectory_plot_by_trial.tex')
MV.trajectory_plot(by="metric", engine="tikz", show=True, save_path=os.getcwd() + '/trajectory_plot_by_metric.tex')
MV.sk_rank_plot(plot_type="box", engine="tikz", show=True, save_path=os.getcwd() + '/sk_rank_plot_box.tex')
MV.sk_rank_plot(plot_type="violin", engine="tikz", show=True, save_path=os.getcwd() + '/sk_rank_plot_violin.tex')
# MV.a12_bar_plot(engine='tikz', show=True)

MV.compile_tikz()



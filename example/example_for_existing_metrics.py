# -*- coding: utf-8 -*-
# file: example.py
# time: 05/02/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import random

import findfile

from metric_visualizer import MetricVisualizer
import numpy as np




trial_num = 5  # number of different trials,
repeat = 20  # number of repeats
metric_num = 5  # number of metrics

#  利用登记好的实验结果绘制图像，可以更改metrics的名字

metric_dict = {
    'metric1': {
        'trial0': [np.random.random() for _ in range(metric_num)],
        'trial1': [np.random.random() for _ in range(metric_num)],
        'trial2': [np.random.random() for _ in range(metric_num)],
        'trial3': [np.random.random() for _ in range(metric_num)],
        'trial4': [np.random.random() for _ in range(metric_num)],
    },
    'metric2': {
        'trial0': [np.random.random() for _ in range(metric_num)],
        'trial1': [np.random.random() for _ in range(metric_num)],
        'trial2': [np.random.random() for _ in range(metric_num)],
        'trial3': [np.random.random() for _ in range(metric_num)],
        'trial4': [np.random.random() for _ in range(metric_num)],
    },
}

MV = MetricVisualizer(name='example',
                      metric_dict=metric_dict,  # 使用登记好的实验结果构建MetricVisualizer
                      trial_tag='Models',
                      trial_tag_list=['model1', 'model2', 'model3', 'model4', 'model5']
                      )

save_prefix = os.getcwd()
MV.summary(save_path=save_prefix, no_print=True)  # save fig into .tex and .pdf format
MV.traj_plot_by_trial(save_path=save_prefix, xlabel='', xrotation=30, minorticks_on=True)  # save fig into .tex and .pdf format
MV.violin_plot_by_trial(save_path=save_prefix)  # save fig into .tex and .pdf format
MV.box_plot_by_trial(save_path=save_prefix)  # save fig into .tex and .pdf format
MV.avg_bar_plot_by_trial(save_path=save_prefix)  # save fig into .tex and .pdf format
MV.sum_bar_plot_by_trial(save_path=save_prefix)  # save fig into .tex and .pdf format
MV.scott_knott_plot(save_path=save_prefix, minorticks_on=False)  # save fig into .tex and .pdf format

print(MV.rank_test_by_trail('trial0'))  # save fig into .tex and .pdf format
print(MV.rank_test_by_metric('metric1'))  # save fig into .tex and .pdf format


MV.dump()  # 手动保存metric-visualizer对象
new_mv = MetricVisualizer.load(findfile.find_cwd_file('.mv'))  # 手动加载metric-visualizer对象
new_mv.traj_plot_by_trial(save_path=save_prefix)  # save fig into .tex and .pdf format

# -*- coding: utf-8 -*-
# file: load_metric_visualizer.py
# time: 21/07/2022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import os

import findfile
from metric_visualizer import MetricVisualizer

mv = MetricVisualizer.load(findfile.find_cwd_file('.mv'))
save_path = os.getcwd()

mv.summary()

mv.traj_plot_by_trial(save_path=save_path)
mv.violin_plot_by_trial(save_path=save_path)
mv.box_plot_by_trial(save_path=save_path)
mv.avg_bar_plot_by_trial(save_path=save_path)
mv.sum_bar_plot_by_trial(save_path=save_path)
mv.scott_knott_plot(save_path=save_path, legend_loc=0)
# mv.rank_test_by_trail()
# mv.rank_test_by_metric()

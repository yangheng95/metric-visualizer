# -*- coding: utf-8 -*-
# file: load_metric_visualizer.py
# time: 21/07/2022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.


import findfile
from metric_visualizer import MetricVisualizer

mv = MetricVisualizer.load(findfile.find_cwd_file('.mv'))
mv.summary()
mv.traj_plot_by_trial()
mv.violin_plot_by_trial()
mv.box_plot_by_trial()
mv.avg_bar_plot_by_trial()
mv.sum_bar_plot_by_trial()
mv.scott_knott_plot()
mv.rank_test_by_trail()
mv.rank_test_by_metric()

# -*- coding: utf-8 -*-
# file: cli_command.py
# time: 06/10/2022 16:59
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2021. All Rights Reserved.
import os

import click
import findfile

from metric_visualizer.metric_visualizer import MetricVisualizer
import multiprocessing

@click.command()
@click.argument('mv')
# @click.option('--save', '-s', default=True, help='Save the figure')
def instant_visualize(mv=None, **kwargs):
    print('Metric Visualizer file: ', mv)
    MV = MetricVisualizer.load(mv)

    MV.summary(dump_path=os.getcwd(), filename='file_name', no_print=False)

    print('Rank test results by trial: ')
    print(MV._rank_test_by_trial(**kwargs))

    print('Rank test results by_metric: ')
    print(MV._rank_test_by_metric(**kwargs))

    # MV.traj_plot_by_trial(xlabel='', xrotation=30, minorticks_on=True)
    # MV.violin_plot_by_trial()
    # MV.box_plot_by_trial()
    # MV.box_plot_by_trial()
    # MV.avg_bar_plot_by_trial()
    # MV.sum_bar_plot_by_trial()

    # MV.traj_plot_by_metric(xlabel='', xrotation=30, minorticks_on=True)
    # MV.violin_plot_by_metric()
    # MV.box_plot_by_metric()
    # MV.box_plot_by_metric()
    # MV.avg_bar_plot_by_metric()
    # MV.sum_bar_plot_by_metric()

    # MV.scott_knott_plot(plot_type='box', minorticks_on=False)
    # MV.scott_knott_plot(plot_type='violin', minorticks_on=False)

    pool = multiprocessing.Pool(os.cpu_count())
    pool.apply_async(MV.traj_plot_by_trial, args=dict(xlabel='', xrotation=30, minorticks_on=True))
    pool.apply_async(MV.violin_plot_by_trial)
    pool.apply_async(MV.box_plot_by_trial)
    pool.apply_async(MV.box_plot_by_trial)
    pool.apply_async(MV.avg_bar_plot_by_trial)
    pool.apply_async(MV.sum_bar_plot_by_trial)

    pool.apply_async(MV.traj_plot_by_metric, args=dict(xlabel='', xrotation=30, minorticks_on=True))
    pool.apply_async(MV.violin_plot_by_metric)
    pool.apply_async(MV.box_plot_by_metric)
    pool.apply_async(MV.box_plot_by_metric)
    pool.apply_async(MV.avg_bar_plot_by_metric)
    pool.apply_async(MV.sum_bar_plot_by_metric)

    pool.apply_async(MV.scott_knott_plot, args=dict(plot_type='box', minorticks_on=False))
    pool.apply_async(MV.scott_knott_plot, args=dict(plot_type='violin', minorticks_on=False))
    pool.close()
    pool.join()

    pool.terminate()

if __name__ == '__main__':
    instant_visualize()

# -*- coding: utf-8 -*-
# file: cli.py
# time: 06/10/2022 16:59
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2021. All Rights Reserved.
import os

import click

from metric_visualizer.metric_visualizer import MetricVisualizer


@click.command()
@click.argument('mv')
@click.option('--save', '-s', default=True, help='Save the figure')
def vis(mv=None, **kwargs):
    print('Metric Visualizer file: ', mv)
    MV = MetricVisualizer.load(mv)

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


if __name__ == '__main__':
    vis()

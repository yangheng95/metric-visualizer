# -*- coding: utf-8 -*-
# file: test_pyabsa.py
# time: 21/07/2022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.

import autocuda
import random

from metric_visualizer import MetricVisualizer

from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

import warnings

device = autocuda.auto_cuda()
warnings.filterwarnings('ignore')

seeds = [random.randint(0, 10000) for _ in range(3)]

max_seq_lens = [60, 70, 80, 90, 100]

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.FAST_LCF_BERT
apc_config_english.lcf = 'cdw'
apc_config_english.max_seq_len = 80
apc_config_english.cache_dataset = False
apc_config_english.patience = 10
apc_config_english.seed = seeds

MV = MetricVisualizer()
apc_config_english.MV = MV

for eta in max_seq_lens:
    apc_config_english.eta = eta
    dataset = ABSADatasetList.Laptop14
    Trainer(config=apc_config_english,
            dataset=dataset,  # train set and test set will be automatically detected
            checkpoint_save_mode=0,  # =None to avoid save model
            auto_device=device  # automatic choose CUDA or CPU
            )
    apc_config_english.MV.next_trial()

save_prefix = '{}_{}'.format(apc_config_english.model_name, apc_config_english.dataset_name)

MV.summary(save_path=save_prefix, no_print=True)  # save fig_preview into .tex and .pdf format
MV.traj_plot_by_trial(save_path=save_prefix, xlabel='', xrotation=30, minorticks_on=True)  # save fig_preview into .tex and .pdf format
MV.violin_plot_by_trial(save_path=save_prefix, xticks=max_seq_lens, xlabel=r'$\eta$')  # save fig_preview into .tex and .pdf format
MV.box_plot_by_trial(save_path=save_prefix, xticks=max_seq_lens, xlabel=r'$\eta$')  # save fig_preview into .tex and .pdf format
MV.avg_bar_plot_by_trial(save_path=save_prefix, xticks=max_seq_lens, xlabel=r'$\eta$')  # save fig_preview into .tex and .pdf format
MV.sum_bar_plot_by_trial(save_path=save_prefix, xticks=max_seq_lens, xlabel=r'$\eta$')  # save fig_preview into .tex and .pdf format
MV.scott_knott_plot(save_path=save_prefix, minorticks_on=False, xticks=max_seq_lens, xlabel=r'$\eta$')  # save fig_preview into .tex and .pdf format

# print(MV.rank_test_by_trail('trial0'))  # save fig_preview into .tex and .pdf format
# print(MV.rank_test_by_metric('metric1'))  # save fig_preview into .tex and .pdf format

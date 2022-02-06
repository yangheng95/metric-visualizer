# MetricVisualizer - for easy managing performance metric

## Install

```bash
pip install metric_visualizer
```

## Usage

If you need to run trial experiments, you can use this tool to make simple plots then fix it manually.

```python3
from metric_visualizer import MetricVisualizer
import numpy as np

MV = MetricVisualizer()

trial_num = 10  # number of different trials, the trial in this repo means controlled experiments
repeat = 10  # number of repeats
metric_num = 3  # number of metrics

for trial in range(trial_num):
    for r in range(repeat):
        t = 0  # metric scale factor        # repeat the experiments to plot violin or box figure
        metrics = [(np.random.random()+n) * 100 for n in range(metric_num)]
        for i, m in enumerate(metrics):
            MV.add_metric('Metric-{}'.format(i + 1), round(m, 2))
    MV.next_trial()

MV.summary(save_path=None)  # plot fig via matplotlib
MV.traj_plot(save_path=None)  # plot fig via matplotlib
MV.violin_plot(save_path=None)  # plot fig via matplotlib
MV.box_plot(save_path=None)  # plot fig via matplotlib
MV.avg_bar_plot(save_path=None)  # plot fig via matplotlib
MV.sum_bar_plot(save_path=None)  # plot fig via matplotlib

save_path = 'example'
MV.traj_plot(save_path=save_path)  #  save fig into tikz and .pdf format
MV.violin_plot(save_path=save_path)  # save fig into tikz and .pdf format
MV.box_plot(save_path=save_path)  # save fig into tikz and .pdf format
MV.avg_bar_plot(save_path=save_path)  # save fig into tikz and .pdf format
MV.sum_bar_plot(save_path=save_path)  # save fig into tikz and .pdf format
```

### Traj Plot

![traj_plot_example](fig/traj_plot_example.png)

### Box Plot

![box_plot_example](fig/box_plot_example.png)

### Violin Plot

![violin_plot_example](fig/violin_plot_example.png)

## Real Usage Example in PyABSA

To analyze the impact of max_seq_len, we can use MetricVisualizer as following:

```bash
pip install pyabsa  # install pyabsa
```

```python3
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

apc_config_english.MV.summary(save_path=None, xticks=max_seq_lens)
apc_config_english.MV.traj_plot(save_path=None, xticks=max_seq_lens)
apc_config_english.MV.violin_plot(save_path=None, xticks=max_seq_lens)
apc_config_english.MV.box_plot(save_path=None, xticks=max_seq_lens)

save_path = '{}_{}'.format(apc_config_english.model_name, apc_config_english.dataset_name)
apc_config_english.MV.summary(save_path=save_path)
apc_config_english.MV.traj_plot(save_path=save_path, xticks=max_seq_lens, xlabel=r'$\eta$')
apc_config_english.MV.violin_plot(save_path=save_path, xticks=max_seq_lens, xlabel=r'$\eta$')
apc_config_english.MV.box_plot(save_path=save_path, xticks=max_seq_lens, xlabel=r'$\eta$')
```

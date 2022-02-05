# MetricVisualizer - for easy managing performance metric

## Install

```bash
pip install metric_visualizer
```

## Usage

If you need to run trail experiments, you can use this tool to make simple plots then fix it manually.

```python3
from metric_visualizer import MetricVisualizer
import numpy as np

MV = MetricVisualizer()

trail_num = 10  # number of different trails
repeat = 10  # number of repeats
metric_num = 3  # number of metrics

for trail in range(trail_num):
    for r in range(repeat):
        t = 0  # metric scale factor        # repeat the experiments to plot violin or box figure
        metrics = [(np.random.random()+n) * 100 for n in range(metric_num)]
        for i, m in enumerate(metrics):
            MV.add_metric('Metric-{}'.format(i + 1), round(m, 2))
    MV.next_trail()

save_path = None
MV.summary(save_path=save_path)  # save fig into .tex and .pdf format
MV.traj_plot(save_path=save_path)  # save fig into .tex and .pdf format
MV.violin_plot(save_path=save_path)  # save fig into .tex and .pdf format
MV.box_plot(save_path=save_path)  # save fig into .tex and .pdf format

save_path = 'example'
MV.traj_plot(save_path=save_path)  # show the fig via matplotlib
MV.violin_plot(save_path=save_path)  # show the fig via matplotlib
MV.box_plot(save_path=save_path)  # show the fig via matplotlib
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

device = autocuda.auto_cuda()

for max_seq_len in max_seq_lens:
    apc_config_english.eta = max_seq_len
    Laptop14 = ABSADatasetList.Laptop14
    Trainer(config=apc_config_english,
            dataset=Laptop14,  # train set and test set will be automatically detected
            checkpoint_save_mode=0,  # =None to avoid save model
            auto_device=device  # automatic choose CUDA or CPU
            )
    apc_config_english.MV.next_trail()

apc_config_english.MV.summary(save_path=None)
apc_config_english.MV.traj_plot(save_path=None, xlabel='Max_Seq_Len')
apc_config_english.MV.violin_plot(save_path=None, xlabel='Max_Seq_Len')
apc_config_english.MV.box_plot(save_path=None, xlabel='Max_Seq_Len')

save_path = '{}_{}'.format(apc_config_english.model_name, apc_config_english.dataset_name)
try:
    apc_config_english.MV.summary(save_path=save_path)
    apc_config_english.MV.traj_plot(save_path=save_path, xlabel='Max_Seq_Len')
    apc_config_english.MV.violin_plot(save_path=save_path, xlabel='Max_Seq_Len')
    apc_config_english.MV.box_plot(save_path=save_path, xlabel='Max_Seq_Len')
except Exception as e:
    pass

```

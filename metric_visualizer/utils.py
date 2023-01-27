# -*- coding: utf-8 -*-
# file: utils.py
# time: 0:05 2023/1/27
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import numpy as np
from scipy import stats


class MetricList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg = np.mean(self)
        self.sum = np.sum(self)
        self.count = len(self)
        self.max = np.max(self)
        self.min = np.min(self)
        self.iqr = np.subtract(*np.percentile(self, [75, 25]))
        self.std = np.std(self)
        self.median = np.median(self)
        self.mode = stats.mode(self)[0][0]
        self.var = np.var(self)

    def add(self, value):
        self.append(value)
        self.update()

    def add_all(self, values):
        self.extend(values)
        self.update()

    def clear(self):
        super().clear()
        self.update()

    def remove(self, value):
        super().remove(value)
        self.update()

    def remove_all(self, values):
        for value in values:
            self.remove(value)
        self.update()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.update()

    def __setslice__(self, i, j, sequence):
        super().__setslice__(i, j, sequence)
        self.update()

    def __delitem__(self, key):
        super().__delitem__(key)
        self.update()

    def __delslice__(self, i, j):
        super().__delslice__(i, j)
        self.update()

    def __iadd__(self, other):
        super().__iadd__(other)
        self.update()

    def update(self):
        self.avg = np.mean(self)
        self.sum = np.sum(self)
        self.count = len(self)
        self.max = np.max(self)
        self.min = np.min(self)
        self.iqr = np.subtract(*np.percentile(self, [75, 25]))
        self.std = np.std(self)
        self.median = np.median(self)
        self.mode = stats.mode(self)[0][0]
        self.var = np.var(self)

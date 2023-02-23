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


class MetricList:
    def __init__(self, *args, **kwargs):
        self.data = list(*args, **kwargs)
        try:
            self.avg = np.nanmean(self.data)
            self.std = np.nanstd(self.data)
            self.median = np.nanmedian(self.data)
            self.mode = stats.mode(self.data, keepdims=True)[0][0]
            self.max = np.nanmax(self.data)
            self.min = np.nanmin(self.data)
            self.var = np.nanvar(self.data)
            self.iqr = stats.iqr(self.data)
            self.skewness = stats.skew(self.data, keepdims=True)
            self.kurtosis = stats.kurtosis(self.data, keepdims=True)
            self.sum = np.nansum(self.data)
            self.count = len(self.data)
        except Exception as e:
            print("Can create MetricList with: ", self.data)

    def _update(self):
        self.avg = np.nanmean(self.data)
        self.std = np.nanstd(self.data)
        self.median = np.nanmedian(self.data)
        self.mode = stats.mode(self.data)[0][0]
        self.max = np.nanmax(self.data)
        self.min = np.nanmin(self.data)
        self.var = np.nanvar(self.data)
        self.iqr = stats.iqr(self.data)
        self.skew = stats.skew(self.data)
        self.kurtosis = stats.kurtosis(self.data)
        self.sum = np.nansum(self.data)
        self.count = len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value
        self._update()

    def __delitem__(self, key):
        del self.data[key]
        self._update()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __reversed__(self):
        return reversed(self.data)

    def __contains__(self, item):
        return item in self.data

    def append(self, item):
        self.data.append(item)
        self._update()

    def extend(self, iterable):
        if isinstance(iterable, MetricList):
            iterable = iterable.data.copy()
        self.data.extend(list(iterable))
        self._update()

    def insert(self, index, item):
        self.data.insert(index, item)
        self._update()

    def pop(self, index=-1):
        self.data.pop(index)
        self._update()

    def remove(self, item):
        self.data.remove(item)
        self._update()

    def clear(self):
        self.data.clear()
        self._update()

    def index(self, item, start=0, stop=None):
        return self.data.index(item, start, stop)

    def count(self, item):
        return self.data.count(item)

    def sort(self, key=None, reverse=False):
        self.data.sort(key, reverse)
        self._update()

    def reverse(self):
        self.data.reverse()
        self._update()

    def copy(self):
        return self.data.copy()

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
        self._data = list(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.avg = np.nanmean(self._data)
        self.std = np.nanstd(self._data)
        self.median = np.nanmedian(self._data)
        self.mode = stats.mode(self._data)[0][0]
        self.max = np.nanmax(self._data)
        self.min = np.nanmin(self._data)
        self.var = np.nanvar(self._data)
        self.iqr = stats.iqr(self._data)
        self.skew = stats.skew(self._data)
        self.kurtosis = stats.kurtosis(self._data)
        self.sum = np.nansum(self._data)
        self.count = len(self._data)

    def _update(self):
        self.avg = np.nanmean(self._data)
        self.std = np.nanstd(self._data)
        self.median = np.nanmedian(self._data)
        self.mode = stats.mode(self._data)[0][0]
        self.max = np.nanmax(self._data)
        self.min = np.nanmin(self._data)
        self.var = np.nanvar(self._data)
        self.iqr = stats.iqr(self._data)
        self.skew = stats.skew(self._data)
        self.kurtosis = stats.kurtosis(self._data)
        self.sum = np.nansum(self._data)
        self.count = len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value
        self._update()

    def __delitem__(self, key):
        del self._data[key]
        self._update()

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __reversed__(self):
        return reversed(self._data)

    def __contains__(self, item):
        return item in self._data

    def append(self, item):
        self._data.append(item)
        self._update()

    def extend(self, iterable):
        self._data.extend(iterable)
        self._update()

    def insert(self, index, item):
        self._data.insert(index, item)
        self._update()

    def pop(self, index=-1):
        self._data.pop(index)
        self._update()

    def remove(self, item):
        self._data.remove(item)
        self._update()

    def clear(self):
        self._data.clear()
        self._update()

    def index(self, item, start=0, stop=None):
        return self._data.index(item, start, stop)

    def count(self, item):
        return self._data.count(item)

    def sort(self, key=None, reverse=False):
        self._data.sort(key, reverse)
        self._update()

    def reverse(self):
        self._data.reverse()
        self._update()

    def copy(self):
        return self._data.copy()

    def __getattr__(self, item):
        return getattr(self._data, item)

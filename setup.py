# -*- coding: utf-8 -*-
# file: setup.py
# time: 2022/2/4
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from setuptools import setup, find_packages

from metric_visualizer import __name__, __version__
from pathlib import Path

cwd = Path(__file__).parent
long_description = (cwd / "README.md").read_text(encoding='utf8')

setup(
    name=__name__,
    version=__version__,
    description='For easy managing performance metric',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yangheng95/metric_visualizer',
    # Author details
    author='Heng, Yang',
    author_email='yangheng@m.scnu.edu.cn',
    python_requires=">=3.6",
    packages=find_packages(),
    include_package_data=True,
    exclude_package_date={'': ['.gitignore']},
    # Choose your license
    license='MIT',
    install_requires=['matplotlib',
                      'tikzplotlib',
                      'findfile',
                      'scipy',
                      'tabulate',
                      'natsort',
                      'numpy',
                      'update_checker',
                      ],
)

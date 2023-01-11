# -*- coding: utf-8 -*-
# file: colalab_example.py
# time: 19:29 2023/1/10
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import os

import findfile
from metric_visualizer.colalab import reformat_tikz_format_for_colalab
from metric_visualizer import MetricVisualizer

if __name__ == "__main__":

    # select a tikz template, paste the path of the template or text of the template
    tex_src_template = r"""
        \documentclass{article}
        \usepackage{pgfplots}
        \usepackage{tikz}
        \usetikzlibrary{intersections}
        \usepackage{helvet}
        \usepackage[eulergreek]{sansmath}

        \begin{document}
        \pagestyle{empty}

        \pgfplotsset{every axis/.append style={
        font = \Large,
        grid = major,
        thick,
        %xmin = 1,
        %xmax = 10,
        %ymin = 3.6,
        %ymax = 4.2,
        line width = 1pt,
        tick style = {line width = 0.8pt}}}

        \begin{tikzpicture}
            \begin{axis}[
                xtick = {1,...,10},
                xticklabels = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1},
                legend pos = south east,
                legend cell align=left,
                legend style={font=\footnotesize},
                legend entries = {r-MOEA/D-STM, R-NSGA-II, g-NSGA-II, r-NSGA-II},
                xlabel = {index},
                ylabel = {R-HV},
            ]
            \addplot[black, mark = *, mark size = 2.5pt] table[x index = 0, y index = 1] {data/delta_hv.dat};
            \addplot[green, mark = square*, mark size = 2.5pt] table[x index = 0, y index = 2] {data/delta_hv.dat};
            \addplot[red, mark = triangle*, mark size = 2.5pt] table[x index = 0, y index = 3] {data/delta_hv.dat}; 
            \end{axis}
        \end{tikzpicture}

        \end{document}
            """

    for tex_file in findfile.find_cwd_files(
        ".tex", exclude_key=[".aux", ".log", ".out", ".gz", ".new"]
    ):
        tex_src_data = tex_file

        style_settings = {
            "legend pos": "north west",
            "legend entries": "{}",
            "xtick": "{}",
            "xticklabels": "{}",
            "ytick": "{}",
            "yticklabels": "{}",
            # write your own style settings here
            # it will be appended to the tikz picture style settings
        }
        reformat_tikz_format_for_colalab(
            tex_src_template,
            tex_src_data,
            output_path=tex_file + ".new.tex",
            style_settings=style_settings,
            # no_legend=True
        )

    for trash in findfile.find_cwd_files(
        or_key=[
            ".aux",
            ".log",
            ".out",
            ".gz",
        ]
    ):
        os.remove(trash)

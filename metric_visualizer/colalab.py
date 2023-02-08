# -*- coding: utf-8 -*-
# file: tex_utils.py
# time: 2:47 2022/12/30
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import os.path
import re
from itertools import zip_longest
from pathlib import Path
from typing import Union

style_dict = {
    "xlabel": r"xlabel.*?\n",
    "ylabel": r"ylabel.*?\n",
    "zlabel": r"zlabel.*?\n",
    "xtick": r"xtick.*?\n",
    "ytick": r"ytick.*?\n",
    "ztick": r"ztick.*?\n",
    "xticklabels": r"xticklabels.*?\n",
    "yticklabels": r"yticklabels.*?\n",
    "zticklabels": r"zticklabels.*?\n",
    "xmin": r"xmin.*?\n",
    "xmax": r"xmax.*?\n",
    "ymin": r"ymin.*?\n",
    "ymax": r"ymax.*?\n",
    "zmin": r"zmin.*?\n",
    "zmax": r"zmax.*?\n",
}


def extract_table_style_from_tex(tex_src_or_file_path):
    """Extract tables from tex file.

    Args:
        tex_src_or_file_path (str): tex file path.

    Returns:
        list: list of tables.
    """
    import re

    if os.path.exists(tex_src_or_file_path):
        with open(tex_src_or_file_path, "r", encoding="utf8") as f:
            tex = f.read()
    else:
        tex = tex_src_or_file_path
    tables = re.findall(r"\\addplot.*?\[.*?};", tex, re.DOTALL)
    styles = [t.split("[")[1].split("]")[0] for t in tables]
    return [tables, styles]


def extract_config_by_name(tex_src_or_file_path, style_name):
    """Extract y label from tex file.

    Args:
        tex_src_or_file_path (str): tex file path.
        style_name (str): style name.

    Returns:
        list: list of y label.
    """
    import re

    if os.path.exists(tex_src_or_file_path):
        with open(tex_src_or_file_path, "r") as f:
            tex = f.read()
    else:
        tex = tex_src_or_file_path
    y_labels = re.findall(style_dict[style_name], tex, re.DOTALL)
    return y_labels[0] if y_labels else None


def extract_legend_from_tex(tex_src_or_file_path):
    """Extract legend from tex file.

    Args:
        tex_src_or_file_path (str): tex file path.

    Returns:
        list: list of legend.
    """
    import re

    if os.path.exists(tex_src_or_file_path):
        with open(tex_src_or_file_path, "r") as f:
            tex = f.read()
    else:
        tex = tex_src_or_file_path
    legends = re.findall(r"\\addlegendentry.*?}", tex, re.DOTALL)
    if legends:
        for i, legend in enumerate(legends):
            legends[i] = legend.replace("\\addlegendentry{", "").replace("}", "")
    if not legends:
        legends = re.findall(r"legend entries.*?}", tex, re.DOTALL)
        if not legends:
            return []
        legends = re.findall(r"{.*?}", legends[0], re.DOTALL)[0][1:-1].split(",")
    return legends


def remove_legend_from_tex(tex_src_or_file_path):
    """Remove legend from tex file.

    Args:
        tex_src_or_file_path (str): tex file path.

    Returns:
        list: list of legend.
    """
    import re

    if os.path.exists(tex_src_or_file_path):
        with open(tex_src_or_file_path, "r") as f:
            tex = f.read()
    else:
        tex = tex_src_or_file_path
    legends = re.findall(r"\\addlegendentry.*?}", tex, re.DOTALL)
    if legends:
        for i, legend in enumerate(legends):
            tex = tex.replace(legend, "")
    if not legends:
        legends = re.findall("legend entries.*?},.*?\n", tex, re.DOTALL)
        if legends:
            for i, legend in enumerate(legends):
                tex = tex.replace(legend, "")
    return tex


def extract_style_from_tex(tex_src_or_file_path):
    """Extract style from tex file.

    Args:
        tex_src_or_file_path (str): tex file path.

    Returns:
        list: list of style.
    """
    import re

    if os.path.exists(tex_src_or_file_path):
        with open(tex_src_or_file_path, "r") as f:
            tex = f.read()
    else:
        tex = tex_src_or_file_path
    styles = re.findall(r"\\begin{axis.*?,.*?\]", tex, re.DOTALL)
    for i, style in enumerate(styles):
        if ",\n]" not in styles[i]:
            styles[i] = styles[i].replace("\n]", ",\n]")
    return styles[0]


def extract_color_from_tex(tex_src_or_file_path):
    """
    Extract color from tex file.

    """
    import re

    if os.path.exists(tex_src_or_file_path):
        with open(tex_src_or_file_path, "r") as f:
            tex = f.read()
    else:
        tex = tex_src_or_file_path
    colors = re.findall(r"\\definecolor{.*?}{RGB}{.*?}", tex, re.DOTALL) + re.findall(
        r"\\definecolor{.*?}{rgb}{.*?}", tex, re.DOTALL
    )
    return colors


def preprocess_style(tex_src_or_file_path):
    """Preprocess style.

    Args:
        tex_src_or_file_path (str): tex file path.

    Returns:
        list: list of style.
    """
    import re

    if os.path.exists(tex_src_or_file_path):
        with open(tex_src_or_file_path, "r") as f:
            tex = f.read()
    else:
        tex = tex_src_or_file_path
    styles = re.findall(r"\\begin{axis.*?,.*?\]", tex, re.DOTALL)
    assert len(styles) == 1
    for i, style in enumerate(styles):
        while "\n " in styles[i]:
            styles[i] = styles[i].replace("\n ", "\n")
        while " \n" in styles[i]:
            styles[i] = styles[i].replace(" \n", "\n")
        while "\t" in styles[i]:
            styles[i] = styles[i].replace("\t", "")

        if ",\n]" not in styles[i]:
            styles[i] = styles[i].replace("\n]", ",\n]")

        # for seg in re.findall(r",\s+]", tex, re.DOTALL):
        #     styles[i] = styles[i].replace(seg, "\n]")

        tex = tex.replace(style, styles[i])

    for seg in re.findall(r"\n\s+\n", tex, re.DOTALL):
        tex = tex.replace(seg, "\n")

    for seg in re.findall(r"\s+=\s+", tex, re.DOTALL):
        tex = tex.replace(seg, "=")

    # leftmargin=0cm,
    lines = tex.split("\n")
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
    tex = "\n".join(lines)

    return tex


def reformat_tikz_format_for_colalab(
    template: Union[str, Path],
    tex_src_to_format: Union[str, Path],
    output_path: Union[str, Path] = None,
    style_settings: dict = None,
    **kwargs,
):
    """Reformat tikz format.

    Args:
        template (str): template, file or tex_text.
        tex_src_to_format (str): tex src to format, file or tex_text.
        output_path (Path): output path.
        style_settings (dict): style settings.

    Returns:
        str: formatted tex src.

    """
    _template = template[:]
    _template = preprocess_style(_template)
    tex_src_to_format = preprocess_style(tex_src_to_format)

    head = re.findall(r"\\begin{tikzpicture}", tex_src_to_format, re.DOTALL)[0]
    for color in extract_color_from_tex(tex_src_to_format):
        _template = _template.replace(head, head + "\n" + color + "\n")

    for new_legend, old_legend in zip_longest(
        extract_legend_from_tex(tex_src_to_format), extract_legend_from_tex(_template)
    ):
        if old_legend and new_legend:
            _template = _template.replace(old_legend, new_legend, 1)
        elif old_legend and not new_legend:
            _template = _template.replace(old_legend, "", 1)
            while ",," in _template:
                _template = _template.replace(",,", ",")
        else:
            style_settings["legend"] = new_legend

    for k, v in style_settings.items():
        _template = _template.replace(
            extract_style_from_tex(_template),
            extract_style_from_tex(_template).replace("]", "{}={},\n]".format(k, v)),
            1,
        )

    if kwargs.get("no_legend", False):
        _template = remove_legend_from_tex(_template)

    new_table_and_styles = extract_table_style_from_tex(tex_src_to_format)
    old_table_and_styles = extract_table_style_from_tex(_template)
    for i in range(max(len(new_table_and_styles[0]), len(old_table_and_styles[0]))):
        new_table, new_style = new_table_and_styles[0][i], new_table_and_styles[1][i]
        if i < len(old_table_and_styles[0]):
            old_table, old_style = (
                old_table_and_styles[0][i],
                old_table_and_styles[1][i],
            )
        else:
            old_table, old_style = "", ""
        if old_table and new_table:
            _template = _template.replace(old_table, new_table, 1)
            _template = _template.replace(old_style, new_style, 1)
        elif old_table and not new_table:
            _template = _template.replace(old_table, "", 1)
        elif not old_table and new_table:
            _template = _template.replace(
                extract_style_from_tex(_template),
                extract_style_from_tex(_template) + "\n" + new_table + "\n",
                1,
            )
            # _template = _template.replace(tikz_plot_style, tikz_plot_style+'\n'+new_table, 1)
        else:
            raise ValueError("old_table and new_table are both None.")

    for k, v in style_dict.items():
        old_style = extract_config_by_name(_template, k)
        new_style = extract_config_by_name(tex_src_to_format, k)
        if old_style and new_style:
            _template = _template.replace(old_style, new_style, 1)
        elif old_style and not new_style:
            _template = _template.replace(old_style, "", 1)
        elif new_style:
            style_settings[k] = new_style.replace(f"{k}=", "").replace(",", "")
            # replace not available, waiting for style setting

    if os.path.exists(tex_src_to_format):
        output_path = os.path.join(Path(tex_src_to_format), ".tex")

    _template = preprocess_style(_template)

    with open(os.path.join(output_path), "w") as f:
        f.write(_template)
    os.system(r'pdflatex "%s"' % output_path)

    os.system(
        r'pdfcrop "%s" "%s"' % (output_path[:-4] + ".pdf", output_path[:-4] + ".pdf")
    )

    return _template

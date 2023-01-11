import numpy as np
import matplotlib.pyplot as plt
import findfile
from metric_visualizer import MetricVisualizer
import numpy as np
from metric_visualizer.colalab import reformat_tikz_format_for_colalab

Cost_r = np.random.random((6, 200))

# b = plt.boxplot(Cost_r.T,
#                  boxprops={'linewidth': '4'},
#                 medianprops={'linewidth': '4', 'color': 'red'},
#                 meanprops={'linewidth': '4'},
#                 capprops={'linewidth': '4'},
#                 flierprops={"marker": "o", "markerfacecolor": "k", "markersize": 10},
#                 whiskerprops={'linewidth': '4'}
#                 )
plt.show()
a = Cost_r[0, :]
b = Cost_r[
    1,
]
c = Cost_r[2, :]
d = Cost_r[3, :]
e = Cost_r[4, :]
f = Cost_r[5, :]
metric_dict = {
    "metric1": {
        "1": list(a),
        "2": list(b),
        "3": list(c),
        "4": list(d),
        "5": list(e),
        "7": list(f),
    }
}

MV = MetricVisualizer(
    name="example",
    metric_dict=metric_dict,  # 使用登记好的实验结果构建MetricVisualizer
    trial_tag="Fid",
    trial_tag_list=["1", "2", "3", "4", "5", "7"],
)

# save_prefix = os.getcwd()
# MV.summary(save_path=save_prefix, no_print=True)  # save fig_preview into .tex and .pdf format
# MV.traj_plot_by_trial(save_path=save_prefix, xlabel='', xrotation=30, minorticks_on=True)  # save fig_preview into .tex and .pdf format
# MV.violin_plot_by_trial(save_path=save_prefix)  # save fig_preview into .tex and .pdf format
# MV.box_plot_by_trial(save_path=save_prefix)  # save fig_preview into .tex and .pdf format
# MV.avg_bar_plot_by_trial(save_path=save_prefix)  # save fig_preview into .tex and .pdf format
# MV.sum_bar_plot_by_trial(save_path=save_prefix)  # save fig_preview into .tex and .pdf format
# MV.A12_bar_plot(save_path=save_prefix)  # save fig_preview into .tex and .pdf format
# MV.scott_knott_plot(save_path=save_prefix, minorticks_on=False)  # save fig_preview into .tex and .pdf format
#
# print(MV.rank_test_by_trail('trial0'))  # save fig_preview into .tex and .pdf format
# print(MV.rank_test_by_metric('metric1'))  # save fig_preview into .tex and .pdf format

# MV.dump()  # 手动保存metric-visualizer对象
# new_mv = MetricVisualizer.load(findfile.find_cwd_file('.mv'))  # 手动加载metric-visualizer对象
# new_mv.traj_plot_by_trial(save_path=save_prefix)  # save fig_preview into .tex and .pdf format

tex_template = r"delta_hv.tex"


style_settings = {
    # "xlabel": "Fid",
    # "ylabel": "ratio",
    # 'xtick': '{0,1,2,3,4,5}',
    # 'xticklabels': '{1,2,3,4,5,7}',
}
reformat_tikz_format_for_colalab(
    template=tex_template,
    tex_src_to_format=MV.box_plot_by_trial(),
    output_path="example.box_plot_by_trial.tikz.new.tex",
    style_settings=style_settings,
)

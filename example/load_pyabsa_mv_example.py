import os

import findfile
from metric_visualizer import MetricVisualizer

metric_dict = {'Max-Test-Acc': {'trial0': [85.12255662426311, 85.3552590753956, 84.88985417313062],
                                'trial1': [85.75860999069191, 85.65001551349674, 85.60347502327025],
                                'trial2': [86.37914986037853, 85.89823146137138, 85.8516909711449],
                                'trial3': [85.3087185851691, 85.8051504809184, 85.63450201675458],
                                'trial4': [85.8051504809184, 85.2156376047161, 86.65839280173752]},
               'Max-Test-F1': {'trial0': [82.32537831851184, 82.58176512035155, 82.05792384083651],
                               'trial1': [83.22186911350632, 82.99446837189276, 82.662652482299],
                               'trial2': [83.74870360929393, 83.2828250939485, 83.17852071805547],
                               'trial3': [82.71561344822, 82.95696529571076, 82.94287174591184],
                               'trial4': [82.76323546386764, 82.34382423002559, 83.97575410001407]}}

MV = MetricVisualizer(name='example', metric_dict=metric_dict)
#
max_seq_lens = [50, 60, 70, 80, 90]
save_prefix = os.getcwd()
MV.summary(save_path=save_prefix, no_print=True)  # save fig_preview into .tex and .pdf format

# # # save fig_preview into .tex and .pdf format
# MV.traj_plot_by_trial(save_path=save_prefix, xlabel='Max modeling length', ylabel='Metric', xticks=max_seq_lens)
# MV.violin_plot_by_trial(save_path=save_prefix, xlabel='Max modeling length', ylabel='Metric', xticks=max_seq_lens)
# MV.box_plot_by_trial(save_path=save_prefix, xlabel='Max modeling length', ylabel='Metric', xticks=max_seq_lens)
# MV.avg_bar_plot_by_trial(save_path=save_prefix, xlabel='Max modeling length', ylabel='Metric', xticks=max_seq_lens)
# MV.sum_bar_plot_by_trial(save_path=save_prefix, xlabel='Max modeling length', ylabel='Metric', xticks=max_seq_lens)

# MV.traj_plot_by_metric(save_path=save_prefix, xlabel='Metric name', ylabel='', xticks=['Max-Test-Acc', 'Max-Test-F1'], legend_label_list=[50, 60, 70, 80, 90])
# MV.violin_plot_by_metric(save_path=save_prefix, xlabel='Metric name', ylabel='', xticks=['Max-Test-Acc', 'Max-Test-F1'], legend_label_list=[50, 60, 70, 80, 90])
# MV.box_plot_by_metric(save_path=save_prefix, xlabel='Metric name', ylabel='', xticks=['Max-Test-Acc', 'Max-Test-F1'], legend_label_list=[50, 60, 70, 80, 90])
# MV.avg_bar_plot_by_metric(save_path=save_prefix, xlabel='Metric name', ylabel='', xticks=['Max-Test-Acc', 'Max-Test-F1'], legend_label_list=[50, 60, 70, 80, 90])
# MV.sum_bar_plot_by_metric(save_path=save_prefix, xlabel='Metric name', ylabel='', xticks=['Max-Test-Acc', 'Max-Test-F1'], legend_label_list=[50, 60, 70, 80, 90])

MV.scott_knott_plot(save_path=save_prefix, xticks=max_seq_lens, xlabel='Max modeling length', minorticks_on=False)
MV.A12_bar_plot(save_path=save_prefix, target_trial=0, xticks=max_seq_lens, xlabel='Max modeling length', minorticks_on=False)
MV.A12_bar_plot(save_path=save_prefix, target_trial=1, xticks=max_seq_lens, xlabel='Max modeling length', minorticks_on=False)

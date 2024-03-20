import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns
sns.set()
plt.rcParams['font.sans-serif'] = 'Times New Roman'
# plt.rcParams['font.sans-serif'] = 'Times New Roman'


# log_path = 'tensorboard log files/'
#
#
# list = [
#     # 'ES, 5 fea, torch_seed=200, other_seed=1, designed fitness func',
#     # 'ES, 5 fea, torch_seed=200, other_seed=2, designed fitness func',
#     # 'ES, 5 fea, torch_seed=200, other_seed=3, designed fitness func',
#     'ES, 5 fea, torch_seed=400, other_seed=1, designed fitness func',
#     'ES, 5 fea, torch_seed=400, other_seed=2, designed fitness func',
#     'ES, 5 fea, torch_seed=400, other_seed=3, designed fitness func',
#     # 'ES, 5 fea, torch_seed=600, other_seed=1, designed fitness func',
#     # 'ES, 5 fea, torch_seed=600, other_seed=2, designed fitness func',
#     # 'ES, 5 fea, torch_seed=600, other_seed=3, designed fitness func',
#     # 'ppo, 5 fea, torch_seed=400, other_seed=1, utilization reward',
#     # 'ppo, 5 fea, torch_seed=400, other_seed=2, utilization reward',
#     # 'ppo, 5 fea, torch_seed=400, other_seed=4, utilization reward',
#     # 'ppo, 5 fea, torch_seed=600, other_seed=1, utilization reward',
#     # 'ppo, 5 fea, torch_seed=600, other_seed=2, utilization reward',
#     # 'ppo, 5 fea, torch_seed=600, other_seed=4, utilization reward',
# ]
#
# # for path in list:
# #     path = log_path + path
# #     ea=event_accumulator.EventAccumulator(path)
# #     ea.Reload()
# #     print(ea.scalars.Keys())
# #     val_psnr=ea.scalars.Items('vali_result')
# #     print(len(val_psnr))
# #
# # for path in list:
# #     path = log_path + path
# #     ea=event_accumulator.EventAccumulator(path)
# #     ea.Reload()
# #     val_psnr=ea.scalars.Items('vali_result')
# #     for i in val_psnr:
# #         print(i.step, i.value)
# label = ["ES", "PPO"]
#
# group_pd_data = []
# df_con_list = []
# for path in list:
#     path = log_path + path
#     ea=event_accumulator.EventAccumulator(path)
#     ea.Reload()
#     val_psnr=ea.scalars.Items('vali_result')
#     step_list = []
#     value_list = []
#
#     for i in val_psnr:
#         step_list.append(i.step)
#         value_list.append(i.value)
#
#     pd_data = {"step":step_list, "makespan":value_list}
#
#     pd_data = pd.DataFrame(pd_data)
#     pd_data["algo"] = "ES"
#     group_pd_data.append(pd_data)
# print("##########################################")
#
# df = pd.concat(group_pd_data, ignore_index=True)
# df_con_list.append(df)
# # pd_data_2 = None
# # for idx, val in enumerate(group_pd_data):
# #     if idx == 0:
# #         pd_data_2 = val
# #     else:
# #         pd_data_2 = pd_data_2 + val
# print("##########################################")
#
# list = [
#     # 'ES, 5 fea, torch_seed=200, other_seed=1, designed fitness func',
#     # 'ES, 5 fea, torch_seed=200, other_seed=2, designed fitness func',
#     # 'ES, 5 fea, torch_seed=200, other_seed=3, designed fitness func',
#     # 'ES, 5 fea, torch_seed=400, other_seed=1, designed fitness func',
#     # 'ES, 5 fea, torch_seed=400, other_seed=2, designed fitness func',
#     # 'ES, 5 fea, torch_seed=400, other_seed=3, designed fitness func',
#     # 'ES, 5 fea, torch_seed=600, other_seed=1, designed fitness func',
#     # 'ES, 5 fea, torch_seed=600, other_seed=2, designed fitness func',
#     # 'ES, 5 fea, torch_seed=600, other_seed=3, designed fitness func',
#     'ppo, 5 fea, torch_seed=400, other_seed=1, utilization reward',
#     'ppo, 5 fea, torch_seed=400, other_seed=2, utilization reward',
#     'ppo, 5 fea, torch_seed=400, other_seed=4, utilization reward',
#     # 'ppo, 5 fea, torch_seed=600, other_seed=1, utilization reward',
#     # 'ppo, 5 fea, torch_seed=600, other_seed=2, utilization reward',
#     # 'ppo, 5 fea, torch_seed=600, other_seed=4, utilization reward',
# ]
#
# group_pd_data = []
# for path in list:
#     path = log_path + path
#     ea=event_accumulator.EventAccumulator(path)
#     ea.Reload()
#     val_psnr=ea.scalars.Items('vali_result')
#     step_list = []
#     value_list = []
#
#     for i in val_psnr:
#         step_list.append(i.step)
#         value_list.append(i.value)
#
#     pd_data = {"step":step_list, "makespan":value_list}
#
#     pd_data = pd.DataFrame(pd_data)
#     pd_data["algo"] = "PPO"
#     group_pd_data.append(pd_data)
# print("##########################################")
#
# df = pd.concat(group_pd_data, ignore_index=True)
# df_con_list.append(df)
#
# print("##########################################")
#
# df_con = pd.concat(df_con_list, ignore_index=True)
#
# sns.lineplot(x="step", y="makespan", hue="algo",data=df_con)
# # plt.title("makespan")
# plt.savefig("./figure.png")
# plt.show()

def smooth(data, weight=0.6):
    last = 0
    smoothed_val = 0
    smoothed = []
    for idx, point in enumerate(data):
        if idx == 0:
            smoothed.append(point)
            last = point
        else:
            try:
                smoothed_val = last * weight + (1-weight)*point
                smoothed.append(smoothed_val)
                last = smoothed_val
            except TypeError:
                print(type(smoothed_val))
                print(type(last))

    return smoothed

def plot_curve(path_list_set, label_list, log_path='tensorboard log files/'):
    df_con_list = []
    for idx, path_list in enumerate(path_list_set):
        group_pd_data = []

        curve_x_len = 1e6
        for path in path_list:
            path = log_path + path
            ea=event_accumulator.EventAccumulator(path)
            ea.Reload()
            val_psnr=ea.scalars.Items('vali_result')
            len_temp = len(val_psnr)
            if len_temp < curve_x_len:
                curve_x_len = len_temp

        for path in path_list:
            path = log_path + path
            ea=event_accumulator.EventAccumulator(path)
            ea.Reload()
            val_psnr=ea.scalars.Items('vali_result')
            step_list = []
            value_list = []

            for i in val_psnr:
                step_list.append(i.step)
                value_list.append(i.value)
            # smooth data
            value_list = smooth(value_list)
            value_list = value_list[:curve_x_len]
            step_list = step_list[:curve_x_len]

            pd_data = {"Epoch":step_list, "Makespan":value_list}

            pd_data = pd.DataFrame(pd_data)
            pd_data["algo"] = label_list[idx]
            group_pd_data.append(pd_data)
            # print("##########################################")

            df = pd.concat(group_pd_data, ignore_index=True)
            df_con_list.append(df)

    df_con = pd.concat(df_con_list, ignore_index=True)

    # print("##########################################")
    ax = sns.lineplot(x="Epoch", y="Makespan", hue="algo", legend="full", data=df_con)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:])
    # plt.ylabel('Average Validation Instances Makespan')

# plt.title("makespan")
#     plt.savefig("./training_curve_(a).svg")
    plt.show()



if __name__=='__main__':
    path_list_set = []
    path_list_1 = [
        # 'ES, 5 fea, torch_seed=200, other_seed=1, designed fitness func',
        # 'ES, 5 fea, torch_seed=200, other_seed=2, designed fitness func',
        # 'ES, 5 fea, torch_seed=200, other_seed=4, designed fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=1, designed fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=2, designed fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=4, designed fitness func',
        '200_f1',
        '200_f2',
        '200_f4',

        # 'ES, 5 fea, torch_seed=200, other_seed=1, exact fitness func',
        # 'ES, 5 fea, torch_seed=200, other_seed=2, exact fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=1, exact fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=2, exact fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=1, exact fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=2, exact fitness func',


        # 'ppo, 5 fea, torch_seed=200, other_seed=1, utilization reward',
        # 'ppo, 5 fea, torch_seed=200, other_seed=2, utilization reward',
        # 'ppo, 5 fea, torch_seed=200, other_seed=4, utilization reward',
        # 'ppo, 5 fea, torch_seed=400, other_seed=1, utilization reward',
        # 'ppo, 5 fea, torch_seed=400, other_seed=2, utilization reward',
        # 'ppo, 5 fea, torch_seed=400, other_seed=4, utilization reward',
        # 'ppo, 5 fea, torch_seed=600, other_seed=1, utilization reward',
        # 'ppo, 5 fea, torch_seed=600, other_seed=2, utilization reward',
        # 'ppo, 5 fea, torch_seed=600, other_seed=4, utilization reward',
    ]
    path_list_set.append(path_list_1)
    path_list_2 = [
        # 'ES, 5 fea, torch_seed=200, other_seed=1, designed fitness func',
        # 'ES, 5 fea, torch_seed=200, other_seed=2, designed fitness func',
        # 'ES, 5 fea, torch_seed=200, other_seed=4, designed fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=1, designed fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=2, designed fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=4, designed fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=1, designed fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=2, designed fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=4, designed fitness func',

        # 'ES, 5 fea, torch_seed=200, other_seed=1, exact fitness func',
        # 'ES, 5 fea, torch_seed=200, other_seed=2, exact fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=1, exact fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=2, exact fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=1, exact fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=2, exact fitness func',


        # 'ppo, 5 fea, torch_seed=200, other_seed=1, utilization reward',
        # 'ppo, 5 fea, torch_seed=200, other_seed=2, utilization reward',
        # 'ppo, 5 fea, torch_seed=200, other_seed=4, utilization reward',
        # 'ppo, 5 fea, torch_seed=400, other_seed=1, utilization reward',
        # 'ppo, 5 fea, torch_seed=400, other_seed=2, utilization reward',
        # 'ppo, 5 fea, torch_seed=400, other_seed=4, utilization reward',
        # 'ppo, 5 fea, torch_seed=600, other_seed=1, utilization reward',
        # 'ppo, 5 fea, torch_seed=600, other_seed=2, utilization reward',
        # 'ppo, 5 fea, torch_seed=600, other_seed=4, utilization reward',
        '400_f1',
        '400_f2',
        '400_f4',
    ]
    # path_list_set.append(path_list_2)
    path_list_3 = [
        # 'ES, 5 fea, torch_seed=200, other_seed=1, designed fitness func',
        # 'ES, 5 fea, torch_seed=200, other_seed=2, designed fitness func',
        # 'ES, 5 fea, torch_seed=200, other_seed=4, designed fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=1, designed fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=2, designed fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=4, designed fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=1, designed fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=2, designed fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=4, designed fitness func',

        # 'ES, 5 fea, torch_seed=200, other_seed=1, exact fitness func',
        # 'ES, 5 fea, torch_seed=200, other_seed=2, exact fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=1, exact fitness func',
        # 'ES, 5 fea, torch_seed=400, other_seed=2, exact fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=1, exact fitness func',
        # 'ES, 5 fea, torch_seed=600, other_seed=2, exact fitness func',


        # 'ppo, 5 fea, torch_seed=200, other_seed=1, utilization reward',
        # 'ppo, 5 fea, torch_seed=200, other_seed=2, utilization reward',
        # 'ppo, 5 fea, torch_seed=200, other_seed=4, utilization reward',
        # 'ppo, 5 fea, torch_seed=400, other_seed=1, utilization reward',
        # 'ppo, 5 fea, torch_seed=400, other_seed=2, utilization reward',
        # 'ppo, 5 fea, torch_seed=400, other_seed=4, utilization reward',
        '600_f1',
        '600_f2',
        '600_f4',
    ]
    # path_list_set.append(path_list_3)
    # label_list = ["ES with proposed fitness function", "ES with exact fitness function", "PPO"]
    label_list = ["200", "400", "600"]
    plot_curve(path_list_set, label_list, log_path='./runs_multiInstance/')


import json
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
import seaborn as sns
import math
import numpy as np
import re
import os
from tabulate import tabulate
from pyvis.network import Network
import streamlit as st
import numpy as np
from typing import List

class TensboardLogAnalyzer:
    """
    Class to analyze TensorBoard logs.
    """

    def __init__(self):
        self.tensorboard_logs = {}
        self.tensorboard_fig = None
    
    def load_tensorboard_logs(self, tensorboard_logs, step_limit=2e6) -> None:
        """
        Load TensorBoard logs from the given paths.
        
        :param tensorboard_logs: Dictionary with model IDs as keys and a list of [model_name, path] as values.
        :param step_limit: Maximum number of steps to consider for the analysis.
        :return: None
        """
        
        # Example:
        # models = {
        #     "druid_v2": ["Regular Masked PPO - Druid", "../../hpc_results/tensorboards/hs_tensorboard/PPO_20/events.out.tfevents.1739542492.tue-gpua001.cluster.169032.0"],
        #     "druid_rnn_large": ["Recurrent Masked PPO - Druid", "../../hpc_results/tensorboards/hs_tensorboard/RecurrentMaskablePPO_4/events.out.tfevents.1739779192.tue-gpua001.cluster.475341.0"],
        #     "druid_rnn_large_retrain_internal": ["Retrained Recurrent Masked PPO - Druid", "../../hpc_results/tensorboards/hs_tensorboard/RecurrentMaskablePPO_9/events.out.tfevents.1740651954.tue-gpua001.cluster.985343.0"],
        #     "general_wide_v3": ["Regular Masked PPO - General", "../../hpc_results/tensorboards/tensorboard_general/PPO_9/events.out.tfevents.1741610481.tue-gpua001.cluster.3798665.0"],
        # }

        #Load rewards
        total_rewards = {}
        steps = {}

        for model_id, model_info in tensorboard_logs.items():
            total_rewards[model_id]=[]
            steps[model_id]=[]
            for e in summary_iterator(model_info[1]):    
                for v in e.summary.value:
                    # Only if step is at most step_limit
                    if e.step > step_limit:
                        break
                    if v.tag == 'rollout/ep_rew_mean':           
                        total_rewards[model_id].append(v.simple_value)
                        steps[model_id].append(e.step)  
                        
        self.tensorboard_logs = tensorboard_logs
        self.total_rewards = total_rewards
        self.steps = steps
        
    
    def plot_rewards(self, smooth_value=0.95) -> plt.Figure:
        """
        Plot the rewards from the TensorBoard logs.
        
        :param smooth_value: Smoothing value for the rewards.
        :return: Matplotlib figure with the rewards plot.
        """

        # Create matplotlib figure with the two lines for each pair of models (druid_v2, druid_rnn_large) and (druid_rnn_large_retrain_internal, general_wide_v3)
        # Font should be 9
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

        colors = sns.color_palette("husl", len(self.tensorboard_logs) + 2)

        smoothed_reward = self.total_rewards.copy()

        # Smooth the rewards
        for model_id, model_info in self.tensorboard_logs.items():
            smoothed_reward[model_id] = self._smooth(self.total_rewards[model_id], smooth_value)

        # Plot the rewards
        for i, (model_id, model_info) in enumerate(self.tensorboard_logs.items()):
            ax.plot(self.steps[model_id], smoothed_reward[model_id], label=model_info[0], color=colors[i-2])
            
        ax.grid()
        ax.set_title("Mean Training Reward vs Steps")
        ax.set_xlabel("Steps (1e6)")
        ax.set_ylabel("Mean Reward")

        # Ticks should be with 1 decimal
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x/1e6)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))

        ax.legend()
        # Set all fontsizes to 9
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(30)

        # Update legend font size
        ax.legend(prop={'size': 20})
        
        return fig
    
    def _smooth(self, scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
        """
        Smooth the given scalars using exponential smoothing.
        
        :param scalars: List of scalars to smooth.
        :param weight: Smoothing weight (between 0 and 1).
        :return: List of smoothed scalars.
        """
        
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
            
        return smoothed
        
        
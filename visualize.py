import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def activityHeatmap(mat_df, title_str="Activity Heatmap"):
    """
    Produce a heatmap of 24 hour activity with midnight centered on the x-axis.
    mat_df: numpy array of shape (Days, 24) representing hourly activity.
    """
    days = mat_df.shape[0]
    
    # mat_df is 0 to 23 hours. We want to center midnight (hour 0).
    # So we take hours 12 to 23, then hours 0 to 11.
    # We can use np.roll to shift the array by 12 hours.
    mat_temp = np.roll(mat_df, shift=12, axis=1)
    
    # Hours labels for x-axis
    hours = np.arange(24)
    x_labels = [(h - 12) % 24 for h in hours]
    
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(mat_temp, cmap='viridis', cbar_kws={'label': 'Activity'})
    
    ax.set_xticks(np.arange(24) + 0.5)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(days) + 0.5)
    ax.set_yticklabels(np.arange(1, days + 1))
    
    plt.title(title_str)
    plt.xlabel("Hour of Day")
    plt.ylabel("Day")
    
    return plt.gcf()

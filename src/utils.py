# src/utils.py

import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import skew, kurtosis

def load_preprocessed_images(file_path):
    """Load preprocessed images from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_dataframe_to_excel(df, file_path):
    """Save a DataFrame to an Excel file."""
    df.to_excel(file_path, index=False)

def plot_statistics_distribution(df, analysis_dir, model, image_type):
    """Plot the distribution of statistics."""
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=df['x_max'], color='skyblue')
    median_x = df['x_max'].median()
    ci_x = sns.utils.ci(df['x_max'], which=95)
    plt.axhline(y=median_x, color='skyblue', linestyle='--', label=f'Median: {median_x:.1f}')
    plt.fill_between(x=[-2, 2], y1=ci_x[0], y2=ci_x[1], color='lightgray', alpha=0.3, label=f'95% CI: [{ci_x[0]:.1f}, {ci_x[1]:.1f}]')
    plt.ylabel('Temperature [°C]', fontsize=16)
    plt.legend(fontsize=14)
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, f'{model}_{image_type}_BFT_Distribution.png'))
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.violinplot(y=df['y_max'], color='salmon')
    median_y = df['y_max'].median()
    ci_y = sns.utils.ci(df['y_max'], which=95)
    plt.axhline(y=median_y, color='salmon', linestyle='--', label=f'Median: {median_y:.1f}')
    plt.fill_between(x=[-2, 2], y1=ci_y[0], y2=ci_y[1], color='lightgray', alpha=0.3, label=f'95% CI: [{ci_y[0]:.1f}, {ci_y[1]:.1f}]')
    plt.ylabel('EEF (%)', fontsize=16)
    plt.legend(fontsize=14)
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, f'{model}_{image_type}_EEF_Distribution.png'))
    plt.show()

    return df['x_max'].std(), df['y_max'].std()


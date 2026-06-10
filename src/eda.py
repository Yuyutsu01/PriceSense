"""
Exploratory Data Analysis (EDA) Module
======================================

Concept:
EDA is a beginner-to-intermediate technique used to understand the structure, characteristics,
and relationships within a dataset. We focus on:
1. Distributions: Inspecting how data values are spread out. For example, housing prices are often right-skewed
   (meaning there are a few very expensive houses pulling the mean up).
2. Outliers: Data points that lie an abnormal distance from other values. For example, houses with huge living area
   but very low prices.
3. Correlations: Measures of linear association. A Pearson correlation of +1 indicates perfect positive correlation,
   0 means no linear correlation, and -1 indicates perfect negative correlation.

Architecture:
- The module contains separate plotting functions for each visual requirement (Price Distribution, Correlation Heatmap,
  and Scatter Plots).
- It creates a dedicated directory `static/plots/` if it does not exist, and saves all plots as high-quality PNGs.
- It calculates numerical diagnostics (skewness, kurtosis, and correlation coefficients) and logs them.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics for clean, modern look
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'figure.titlesize': 14,
    'axes.titlesize': 12
})

def plot_price_distribution(df: pd.DataFrame, output_dir: str = "static/plots") -> None:
    """
    Plots and saves the distribution of housing prices (SalePrice) along with skewness and kurtosis.
    """
    os.makedirs(output_dir, exist_ok=True)
    target = 'SalePrice'
    
    if target not in df.columns:
        print(f"Error: {target} column not found in DataFrame for Price Distribution plot.")
        return
        
    prices = df[target].dropna()
    skewVal = prices.skew()
    kurtVal = prices.kurt()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    # Using HSL/modern color palette (Sleek Indigo: #58508d)
    sns.histplot(prices, kde=True, color='#58508d', bins=40, ax=ax, edgecolor='white', alpha=0.85)
    
    ax.set_title("Housing Sale Price Distribution", fontsize=14, pad=15, fontweight='bold', color='#222222')
    ax.set_xlabel("Sale Price ($)", labelpad=10, fontsize=11)
    ax.set_ylabel("Frequency", labelpad=10, fontsize=11)
    
    # Format x-axis labels with commas (e.g. 100,000)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
    
    # Text annotation for Skewness and Kurtosis
    textstr = '\n'.join((
        f"Skewness: {skewVal:.2f}",
        f"Kurtosis: {kurtVal:.2f}"
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
            
    plt.tight_layout()
    output_path = os.path.join(output_dir, "price_distribution.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved Price Distribution plot to {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "static/plots") -> None:
    """
    Plots and saves a correlation heatmap for key numerical features and the target variable.
    """
    os.makedirs(output_dir, exist_ok=True)
    cols = ['LotArea', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'SalePrice']
    
    # Filter columns to only those present in df
    available_cols = [c for c in cols if c in df.columns]
    if len(available_cols) < 2:
        print("Error: Not enough numerical columns available to compute correlation.")
        return
        
    corr = df[available_cols].corr()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    # Modern harmonized colormap (Cool Warm divergency)
    sns.heatmap(
        corr, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=1, 
        ax=ax,
        cbar_kws={'label': 'Pearson Correlation Coeff'},
        annot_kws={'size': 11, 'weight': 'semibold'}
    )
    
    ax.set_title("Correlation Heatmap: Key Features & Price", fontsize=13, pad=15, fontweight='bold', color='#222222')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved Correlation Heatmap plot to {output_path}")


def plot_scatter_plots(df: pd.DataFrame, output_dir: str = "static/plots") -> None:
    """
    Plots scatter plots of main features vs SalePrice in a 2x2 grid layout to identify outliers and trends.
    """
    os.makedirs(output_dir, exist_ok=True)
    features = ['LotArea', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']
    target = 'SalePrice'
    
    if target not in df.columns:
        print("Error: SalePrice target variable missing for Scatter Plots.")
        return
        
    available_features = [f for f in features if f in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
    
    for i, feature in enumerate(available_features):
        if i >= len(axes):
            break
        ax = axes[i]
        sns.scatterplot(data=df, x=feature, y=target, alpha=0.6, color=colors[i % len(colors)], ax=ax)
        
        # Fit a trendline to help visualize linear relationships
        valid_data = df[[feature, target]].dropna()
        if len(valid_data) > 0:
            sns.regplot(data=valid_data, x=feature, y=target, scatter=False, color='#222222', ax=ax,
                        line_kws={'linestyle': '--', 'linewidth': 1.2})
            
        ax.set_title(f"{feature} vs Sale Price", fontsize=11, fontweight='bold')
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Sale Price ($)", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x/1000)}k"))
        
        # Format LotArea with commas if plotting LotArea
        if feature == 'LotArea':
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
            
    # Remove unused subplots
    for j in range(len(available_features), len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Key Numerical Features vs. House Sale Price", fontsize=14, fontweight='bold', color='#111111', y=0.98)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "scatter_plots.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved Scatter Plots to {output_path}")


def run_eda(df: pd.DataFrame, output_dir: str = "static/plots") -> None:
    """
    Helper function to run the full Exploratory Data Analysis suite.
    """
    print("Starting Exploratory Data Analysis...")
    plot_price_distribution(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_scatter_plots(df, output_dir)
    print("EDA completed successfully.")

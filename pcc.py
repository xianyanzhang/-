# 安装必要库（在 VSCode 终端中运行）
# pip install pandas matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# SCI 论文图形全局设置（英文标签）
# ---------------------------
plt.rcParams.update({
    # 单栏宽度 3.5 inch × 3.5 inch
    'figure.figsize': (3.5, 3.5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'text.usetex': False,
    'figure.autolayout': True
})

# ---------------------------
# 数据加载函数（英文列名）
# ---------------------------
def load_final_data(filepath):
    """Load and clean data with English column names."""
    column_names = [
        'Max', 'Min', 'Mean', 'StdDev',
        'Rainfall_mm', 'Temperature_C', 'Evaporation_mm',
        'Sunshine_h', 'WindSpeed_mps'
    ]
    def advanced_clean(x):
        try:
            return float(str(x).strip('"').replace(',', '.').strip())
        except:
            return np.nan

    df = pd.read_csv(
        filepath,
        header=None,
        names=column_names,
        converters={i: advanced_clean for i in range(len(column_names))},
        encoding='utf-8',
        engine='python',
        na_values=['', 'NA', 'NaN']
    )
    # 简单验证
    assert df.shape[1] == len(column_names), f"Expected {len(column_names)} columns, got {df.shape[1]}"
    assert not df.isnull().values.any(), "Found NaNs; please check the data file"
    return df

# ---------------------------
# 主程序
# ---------------------------
if __name__ == "__main__":
    # 1. Load data
    df = load_final_data('整体相关性分析.csv')

    # 2. Define groups
    resi_cols    = ['Max', 'Min', 'Mean', 'StdDev']
    climate_cols = ['Rainfall_mm', 'Temperature_C', 'Evaporation_mm', 'Sunshine_h', 'WindSpeed_mps']

    # 3. Compute correlation matrix
    corr_matrix = df[resi_cols + climate_cols].corr().loc[resi_cols, climate_cols]

    # 4. Plot heatmap
    plt.figure()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='RdYlBu_r',
        center=0,
        linewidths=0.8,
        linecolor='white',
        annot_kws={"size": 8},
        cbar_kws={'label': 'Pearson Correlation Coefficient'},
        mask=np.abs(corr_matrix) < 0.01
    )

    # 5. Labels
    plt.xlabel('Climate Variables', labelpad=10)
    plt.ylabel('RESI Metrics', labelpad=10)
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)

    # 6. Save high-quality outputs
    plt.savefig('advanced_correlation_analysis.pdf', bbox_inches='tight')
    plt.savefig('advanced_correlation_analysis.png', bbox_inches='tight')

    # 7. Show
    plt.show()

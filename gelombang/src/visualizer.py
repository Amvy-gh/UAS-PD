from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style='whitegrid')

class Visualizer:
    def __init__(self, df_labeled: pd.DataFrame):
        self.df = df_labeled.copy()

    def plot_outlier_steps(self, x_col: str = 'WindSpeed(knots)', y_col: str = 'Hsig(m)', sample_n: int = 2000, outpath: str = 'Outlier_Steps.png'):
        # random sample to avoid biased first-rows
        if len(self.df) > sample_n:
            df_sample = self.df.sample(sample_n, random_state=42)
        else:
            df_sample = self.df.copy()

        plt.figure(figsize=(15, 5))

        # 1. Raw
        plt.subplot(1, 3, 1)
        plt.title('1. Data Awal (Raw)')
        sns.scatterplot(data=df_sample, x=x_col, y=y_col, alpha=0.5, s=15)
        plt.xlabel(f"{x_col} (scaled)")
        plt.ylabel(f"{y_col} (scaled)")

        # 2. Detected
        plt.subplot(1, 3, 2)
        plt.title('2. Hasil Deteksi iForest')
        sns.scatterplot(data=df_sample, x=x_col, y=y_col, hue='is_outlier', palette={0:'blue',1:'red'}, alpha=0.6, s=15)

        # 3. Cleaned
        df_clean = df_sample[df_sample['is_outlier'] == 0]
        plt.subplot(1, 3, 3)
        plt.title('3. Setelah Outlier Dibuang')
        sns.scatterplot(data=df_clean, x=x_col, y=y_col, alpha=0.6, s=15)

        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        print(f" - Saved plot to {outpath}")
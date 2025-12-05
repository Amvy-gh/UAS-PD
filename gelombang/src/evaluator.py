from typing import Tuple, Dict
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 5. CLASS EVALUATOR
# ==========================================
class Evaluator:
    """Evaluate classification performance improvement (Clean vs Dirty)."""
    
    def __init__(self, df_labeled: pd.DataFrame, target_col: str = 'Location'):
        self.df = df_labeled.copy()
        self.target_col = target_col

    def evaluate(self):
        print('\n' + '='*60)
        print(f"⚖️  [5] EVALUASI PERFORMA (KLASIFIKASI)")
        print('='*60)

        # Hapus kolom bantuan yang tidak boleh masuk training
        cols_to_drop = ['is_outlier', 'anomaly_score', self.target_col]
        
        # 1. Dataset Kotor (Raw)
        X_dirty = self.df.drop(columns=cols_to_drop)
        y_dirty = self.df[self.target_col]

        # 2. Dataset Bersih (Clean)
        df_clean = self.df[self.df['is_outlier'] == 0]
        X_clean = df_clean.drop(columns=cols_to_drop)
        y_clean = df_clean[self.target_col]
        
        # 3. Train & Evaluate
        metrics_dirty, model_dirty, ytest_dirty, preds_dirty = self._train_predict(X_dirty, y_dirty, "DATA KOTOR")
        metrics_clean, model_clean, ytest_clean, preds_clean = self._train_predict(X_clean, y_clean, "DATA BERSIH")
        
        # 4. Tampilkan Tabel Perbandingan
        print("\n🏆 KESIMPULAN AKHIR (PERBANDINGAN METRIK):")
        print("-" * 75)
        print(f"{'METRIC':<15} | {'RAW DATA':<12} | {'CLEAN DATA':<12} | {'IMPROVEMENT':<12}")
        print("-" * 75)
        for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            val_dirty = metrics_dirty[m]
            val_clean = metrics_clean[m]
            imp = val_clean - val_dirty
            sign = "+" if imp > 0 else ""
            print(f"{m:<15} | {val_dirty:.4f}       | {val_clean:.4f}       | {sign}{imp:.4f}")
        print("-" * 75)

        # 5. Generate Visualizations (Confusion Matrix & Decision Tree)
        self._plot_results(ytest_dirty, preds_dirty, ytest_clean, preds_clean, model_dirty, model_clean, X_dirty.columns)

    def _train_predict(self, X, y, label):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Batasi kedalaman pohon agar tidak overfitting & lebih mudah divisualisasi
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        # Hitung Metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, preds, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, preds, average='weighted', zero_division=0)
        }
        
        return metrics, clf, y_test, preds

    def _plot_results(self, y_dirty, p_dirty, y_clean, p_clean, m_dirty, m_clean, feature_names):
        # A. Confusion Matrix
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.heatmap(confusion_matrix(y_dirty, p_dirty), annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title("Confusion Matrix - DIRTY Data")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        
        sns.heatmap(confusion_matrix(y_clean, p_clean), annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title("Confusion Matrix - CLEAN Data")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
        
        plt.tight_layout()
        if not os.path.exists('output'): os.makedirs('output')
        plt.savefig("output/conf_matrices.png", dpi=150)
        print("✅ Confusion Matrices disimpan ke: output/conf_matrices.png")

        # B. Visualisasi Pohon Keputusan (Tree Plot)
        from sklearn.tree import plot_tree
        fig, axes = plt.subplots(2, 1, figsize=(24, 16))
        
        plot_tree(m_dirty, ax=axes[0], filled=True, feature_names=feature_names, fontsize=8, max_depth=3)
        axes[0].set_title("Decision Tree Structure - DIRTY Data (Top 3 Levels)", fontsize=14, fontweight='bold')
        
        plot_tree(m_clean, ax=axes[1], filled=True, feature_names=feature_names, fontsize=8, max_depth=3)
        axes[1].set_title("Decision Tree Structure - CLEAN Data (Top 3 Levels)", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig("output/decision_trees.png", dpi=150)
        print("✅ Decision Tree Plots disimpan ke: output/decision_trees.png")
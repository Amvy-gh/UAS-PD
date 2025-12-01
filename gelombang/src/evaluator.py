from typing import Tuple
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    def __init__(self, df_labeled: pd.DataFrame, target_col: str = 'Location'):
        self.df = df_labeled.copy()
        self.target_col = target_col

    def _train_dt(self, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> float:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
        dt = DecisionTreeClassifier(max_depth=7, random_state=random_state)
        dt.fit(X_train, y_train)
        preds = dt.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return acc, y_test, preds

    def compare_dirty_clean(self) -> dict:
        # Dirty
        X_dirty = self.df.drop(columns=[self.target_col, 'is_outlier'])
        y_dirty = self.df[self.target_col]

        # Clean
        df_clean = self.df[self.df['is_outlier'] == 0]
        X_clean = df_clean.drop(columns=[self.target_col, 'is_outlier'])
        y_clean = df_clean[self.target_col]

        acc_dirty, ytest_dirty, preds_dirty = self._train_dt(X_dirty, y_dirty)
        acc_clean, ytest_clean, preds_clean = self._train_dt(X_clean, y_clean)

        improvement = acc_clean - acc_dirty

        print(f"Akurasi Data Kotor: {acc_dirty*100:.2f}%")
        print(f"Akurasi Data Bersih: {acc_clean*100:.2f}%")
        print(f"Peningkatan: {improvement*100:.2f}%")

        # Save confusion matrices for further inspection
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(confusion_matrix(ytest_dirty, preds_dirty), annot=True, fmt='d', ax=axes[0])
        axes[0].set_title('Confusion Matrix - Dirty')
        sns.heatmap(confusion_matrix(ytest_clean, preds_clean), annot=True, fmt='d', ax=axes[1])
        axes[1].set_title('Confusion Matrix - Clean')
        plt.tight_layout()
        plt.savefig('conf_matrices.png', dpi=150)
        print(' - Confusion matrices saved to conf_matrices.png')

        return {
            'acc_dirty': acc_dirty,
            'acc_clean': acc_clean,
            'improvement': improvement,
            'ytest_dirty': ytest_dirty,
            'preds_dirty': preds_dirty,
            'ytest_clean': ytest_clean,
            'preds_clean': preds_clean,
        }
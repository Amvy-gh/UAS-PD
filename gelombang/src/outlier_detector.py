from typing import Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class OutlierDetector:
    def __init__(self, df: pd.DataFrame, target_col: str = 'Location'):
        self.df = df.copy()
        self.target_col = target_col
        self.model: Optional[IsolationForest] = None

    def detect(
        self, 
        n_estimators: int = 300,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Jurnal cyber tidak menggunakan contamination tetap.
        Threshold ditentukan dari distribusi anomaly score:
        
        threshold = mean(score) + 2 * std(score)
        """
        X = self.df.drop(columns=[self.target_col])

        # train model tanpa contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination='auto',   # auto hanya untuk decision_function, bukan mark outlier
            random_state=random_state,
            n_jobs=-1
        )

        # fit model
        self.model.fit(X)

        # anomaly score (semakin tinggi semakin aneh)
        scores = -self.model.decision_function(X)
        self.df['anomaly_score'] = scores

        # --- Sesuai jurnal: adaptive threshold ---
        mu = scores.mean()
        sigma = scores.std()
        threshold = mu + 2 * sigma

        print(f" - Adaptive Threshold (μ + 2σ): {threshold:.4f}")

        # outlier: score di atas threshold
        self.df['is_outlier'] = (scores > threshold).astype(int)

        # laporan jumlah
        num_outliers = int(self.df['is_outlier'].sum())
        print(f" - Detected {num_outliers} outliers ({num_outliers / len(self.df) * 100:.2f}%)")

        return self.df

    def remove_outliers(self) -> pd.DataFrame:
        return self.df[self.df['is_outlier'] == 0].copy()

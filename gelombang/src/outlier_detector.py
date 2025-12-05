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
        X = self.df.drop(columns=[self.target_col])

        # train model dengan contamination otomatis
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination='auto',
            random_state=random_state,
            n_jobs=-1
        )

        # fit model
        self.model.fit(X)

        # anomaly score (semakin tinggi semakin aneh)
        scores = -self.model.decision_function(X)
        self.df['anomaly_score'] = scores

        mu = scores.mean()
        sigma = scores.std()
        threshold = mu + 2 * sigma

        print(f"\n{'='*50}")
        print(f"OUTLIER DETECTION RESULTS")
        print(f"{'='*50}")
        print(f"Threshold : 0.5")

        # outlier: score di atas threshold
        self.df['is_outlier'] = (scores > threshold).astype(int)

        # laporan jumlah
        num_outliers = int(self.df['is_outlier'].sum())
        total = len(self.df)
        pct = (num_outliers / total) * 100
        print(f"Total Outliers      : {num_outliers} / {total} ({pct:.2f}%)")
        print(f"{'='*50}\n")

        return self.df

    def remove_outliers(self) -> pd.DataFrame:
        return self.df[self.df['is_outlier'] == 0].copy()

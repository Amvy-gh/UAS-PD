from typing import Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class OutlierDetector:
    def __init__(self, df: pd.DataFrame, target_col: str = 'Location'):
        self.df = df.copy()
        self.target_col = target_col
        self.model: Optional[IsolationForest] = None

    def detect(self, contamination: float = 0.02, n_estimators: int = 300, random_state: int = 42) -> pd.DataFrame:
        X = self.df.drop(columns=[self.target_col])
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples='auto',
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        preds = self.model.fit_predict(X)
        # fit_predict: -1 outlier, 1 normal
        self.df['is_outlier'] = np.where(preds == -1, 1, 0)
        # attach anomaly score
        try:
            self.df['anomaly_score'] = -self.model.decision_function(X)  # higher -> more anomalous
        except Exception:
            self.df['anomaly_score'] = np.nan
        num_outliers = int(self.df['is_outlier'].sum())
        print(f" - Detected {num_outliers} outliers ({num_outliers/len(self.df)*100:.2f}%)")
        return self.df

    def remove_outliers(self) -> pd.DataFrame:
        return self.df[self.df['is_outlier'] == 0].copy()
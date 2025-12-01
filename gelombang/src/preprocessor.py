from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE


class Preprocessor:
    """Preprocessing pipeline inspired by the Ripan et al. (iForest paper).

    Key steps:
    - handle missing values (drop small fraction or impute)
    - label-encode categorical features
    - standard scale features
    - optional feature selection via RFE + ExtraTrees
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'Location'):
        self.df = df.copy()
        self.target_col = target_col
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.selected_features: Optional[List[str]] = None

    def handle_missing(self, drop_threshold: float = 0.01) -> None:
        # Drop rows where target is null
        before = len(self.df)
        self.df = self.df.dropna(subset=[self.target_col])
        after = len(self.df)
        print(f" - Drop NaN in target: {before} -> {after}")

        # For other columns, follow simple heuristic
        missing = self.df.isnull().sum()
        total = len(self.df)
        for col, cnt in missing.items():
            if cnt == 0 or col == self.target_col:
                continue
            frac = cnt / total
            if frac <= drop_threshold:
                # drop rows containing NA for that col
                self.df = self.df.dropna(subset=[col])
                print(f" - Dropped {col} rows with NA (small fraction)")
            else:
                # impute
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    imp = SimpleImputer(strategy='median')
                else:
                    imp = SimpleImputer(strategy='most_frequent')
                self.df[col] = imp.fit_transform(self.df[[col]])
                print(f" - Imputed {col} with {imp.strategy}")

    def remove_redundant_columns(self) -> None:
        cols_to_drop = [c for c in self.df.columns if '(Scale)' in c or '(compass)' in c]
        if 'Time(UTC/GMT)' in self.df.columns:
            cols_to_drop.append('Time(UTC/GMT)')
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
            print(f" - Dropped redundant cols: {cols_to_drop}")

    def encode_categoricals(self) -> None:
        for col in self.df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f" - Label encoded: {col}")

    def scale_features(self, exclude_target: bool = True) -> None:
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if exclude_target and self.target_col in num_cols:
            num_cols.remove(self.target_col)
        if not num_cols:
            return
        self.df[num_cols] = self.scaler.fit_transform(self.df[num_cols])
        print(f" - Standard scaled numeric features ({len(num_cols)} cols)")

    def select_features_rfe(self, n_features: int = 15, random_state: int = 42) -> None:
        """Use ExtraTreesClassifier + RFE to pick top features. Requires target present."""
        if self.target_col not in self.df.columns:
            raise ValueError("Target column not found for feature selection")

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        etc = ExtraTreesClassifier(n_estimators=100, random_state=random_state)
        rfe = RFE(estimator=etc, n_features_to_select=min(n_features, X.shape[1]), step=0.1)
        rfe.fit(X, y)
        mask = rfe.support_
        selected = X.columns[mask].tolist()
        self.selected_features = selected
        print(f" - Selected {len(selected)} features via RFE")

    def get_processed(self, use_selected: bool = False) -> pd.DataFrame:
        if use_selected and self.selected_features is not None:
            cols = self.selected_features + [self.target_col]
            return self.df[cols].copy()
        return self.df.copy()
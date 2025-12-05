from typing import Optional, Tuple
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, filepath: str, target_col: str = 'Location'):
        self.filepath = filepath
        self.target_col = target_col
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        return self.df

    def inspect(self, show_top: int = 10) -> dict:
        if self.df is None:
            raise ValueError("Data not loaded. Call .load() first.")

        print('\n=== [1] INSPEKSI DATA AWAL ===')
        shape = self.df.shape
        print(f"Shape: {shape}")

        missing = self.df.isnull().sum()
        missing_nonzero = missing[missing > 0]
        print('\nMissing Values (top):')
        print(missing_nonzero.head(show_top))

        zeros = {}
        for col in ['Hsig(m)', 'WindSpeed(knots)']:
            if col in self.df.columns:
                zeros[col] = int((self.df[col] == 0).sum())
        if zeros:
            print('\nZero counts (suspicious columns):')
            for k, v in zeros.items():
                print(f" - {k}: {v} rows")

        dtypes = self.df.dtypes
        print('\nDtypes summary:')
        print(dtypes.value_counts())

        # Provide recommendation for handling missing values
        recommendation = {}
        total_rows = len(self.df)
        for col, cnt in missing_nonzero.items():
            frac = cnt / total_rows
            if frac < 0.01:
                recommendation[col] = 'drop_rows'  # small fraction -> drop
            else:
                # numerical -> median, categorical -> mode
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    recommendation[col] = 'impute_median'
                else:
                    recommendation[col] = 'impute_mode'

        # If zeros large fraction for Hsig(m) assume sensor error -> treat as suspicious
        if 'Hsig(m)' in zeros and zeros['Hsig(m)'] > 0:
            recommendation['Hsig(m)'] = 'treat_as_suspicious_outlier_candidate'

        return {
            'shape': shape,
            'missing_counts': missing_nonzero,
            'zero_counts': zeros,
            'dtypes': dtypes,
            'recommendation': recommendation,
        }
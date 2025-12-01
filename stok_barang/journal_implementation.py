import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seed for reproducibility
np.random.seed(42)

def main():
    print("Loading data...")
    file_path = 'd:/TesTugas/DATMIN/hasil_perbaikan.csv'
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # --- 1. Data Pre-processing (Data Preparation) ---
    print("Preprocessing data...")
    
    # Convert Date
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    
    # Sort for Stock Calculation (Ground Truth)
    df = df.sort_values(by=['KODE', 'TANGGAL', 'NO_TRANSAKSI'])
    
    # Calculate Stock to generate Ground Truth (Target)
    # We assume the goal is to detect transactions where QTY_KLR > Available Stock
    print("Generating Ground Truth (Stock Logic)...")
    
    # Group by Product
    # Calculate cumulative In and Out
    df['cum_msk'] = df.groupby('KODE')['QTY_MSK'].cumsum()
    df['cum_klr'] = df.groupby('KODE')['QTY_KLR'].cumsum()
    
    # Stock available for the current transaction (before removing current QTY_KLR)
    # Stock_After_Previous = (Cum_Msk_Prev - Cum_Klr_Prev)
    # Current_Available = Stock_After_Previous + Current_QTY_MSK
    # Simplified: Current_Available = Cum_Msk - (Cum_Klr - Current_QTY_KLR)
    df['stock_available'] = df['cum_msk'] - (df['cum_klr'] - df['QTY_KLR'])
    
    # Define Ground Truth: 1 if Outlier (Qty Out > Available), 0 if Normal
    df['is_outlier'] = (df['QTY_KLR'] > df['stock_available']).astype(int)
    
    print(f"Total Outliers (Ground Truth): {df['is_outlier'].sum()} out of {len(df)}")
    
    # Encoding Categorical Features
    # Journal: "Label Encoding ... to avoid excessive dimensions"
    le = LabelEncoder()
    categorical_cols = ['NAMA_PRODUK', 'UNIT'] # KODE is redundant with NAMA_PRODUK
    
    for col in categorical_cols:
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
    # Scaling
    # Journal: "Standard Scaler"
    scaler = StandardScaler()
    numeric_cols = ['QTY_MSK', 'NILAI_MSK', 'QTY_KLR', 'NILAI_KLR']
    
    # We work on a copy for features to avoid messing up original df for display
    X_processed = df.copy()
    X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
    
    # Define Features for Model
    feature_cols = numeric_cols + [f'{col}_encoded' for col in categorical_cols]
    X = X_processed[feature_cols]
    y = df['is_outlier']
    
    # --- 2. Important Feature Extraction (RFE) ---
    print("Performing Feature Selection (RFE)...")
    # Journal: "RFE with Extra Tree Classifier... select 15 features"
    # We have fewer than 15 features, so we will select top k (e.g., 4) or just use all if few.
    # Let's try to select top 4 features to demonstrate the step.
    
    estimator = ExtraTreesClassifier(n_estimators=50, random_state=42)
    selector = RFE(estimator, n_features_to_select=4, step=1)
    selector = selector.fit(X, y)
    
    selected_features = [f for f, s in zip(feature_cols, selector.support_) if s]
    print(f"Selected Features: {selected_features}")
    
    X_selected = X[selected_features]
    
    # --- 3. Outlier Detection with Isolation Forest ---
    print("Running Isolation Forest...")
    # Journal: "Build iTrees... Split... Isolation Logic"
    
    # Contamination: The journal doesn't specify, but usually it's the expected % of outliers.
    # We can use the ground truth percentage or a small value.
    contamination_rate = df['is_outlier'].mean()
    if contamination_rate == 0:
        contamination_rate = 0.01 # Fallback
    
    print(f"Contamination Rate: {contamination_rate:.4f}")
    
    iso_forest = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42, n_jobs=-1)
    iso_forest.fit(X_selected)
    
    # Predictions: 1 for inlier, -1 for outlier
    preds = iso_forest.predict(X_selected)
    
    # Scores: lower is more abnormal (sklearn default).
    # Journal: "Score close to 1 is outlier, < 0.5 is normal".
    # Sklearn's score_samples returns negative values (offset).
    # decision_function returns values where < 0 is outlier.
    # We will stick to the binary prediction for evaluation.
    
    df['iforest_pred'] = preds
    # Map to 0 (Normal) and 1 (Outlier)
    df['pred_label'] = df['iforest_pred'].apply(lambda x: 1 if x == -1 else 0)
    
    # Remove predicted outliers
    df_clean = df[df['iforest_pred'] == 1] # Keep inliers
    remaining_errors = df_clean['is_outlier'].sum()

    # --- Output to File ---
    with open('results.txt', 'w') as f:
        f.write("--- Implementation Results ---\n")
        f.write(f"Total Outliers (Ground Truth): {df['is_outlier'].sum()} out of {len(df)}\n")
        f.write(f"Selected Features: {selected_features}\n")
        f.write(f"Contamination Rate: {contamination_rate:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y, df['pred_label'])))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y, df['pred_label']))
        f.write("\n\n--- Step 4: Outlier Removal & Classification Simulation ---\n")
        f.write(f"Original size: {len(df)}, Cleaned size: {len(df_clean)}\n")
        f.write(f"Removed {len(df) - len(df_clean)} instances.\n")
        f.write(f"Remaining Stock Errors in Cleaned Dataset: {remaining_errors}\n")
        
    print("Results written to results.txt")

    # --- Visualizations ---
    print("Generating Visualizations...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, df['pred_label'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Normal', 'Predicted Outlier'],
                yticklabels=['Actual Normal', 'Actual Outlier'])
    plt.title('Confusion Matrix: Stock Logic vs Isolation Forest')
    plt.ylabel('Actual (Ground Truth)')
    plt.xlabel('Predicted (Isolation Forest)')
    plt.tight_layout()
    plt.savefig('viz_confusion_matrix.png')
    plt.close()
    
    # 2. Visualization Before vs After Detection (Scatter Plot)
    # We use NILAI_KLR vs QTY_KLR as they are likely correlated and significant
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Ground Truth (What actually are errors)
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='QTY_KLR', y='NILAI_KLR', hue='is_outlier', palette={0: 'blue', 1: 'red'}, alpha=0.6)
    plt.title('Ground Truth: Actual Stock Errors')
    plt.legend(title='Is Error', labels=['Normal', 'Error'])
    
    # Subplot 2: iForest Detection (What model found)
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='QTY_KLR', y='NILAI_KLR', hue='pred_label', palette={0: 'blue', 1: 'red'}, alpha=0.6)
    plt.title('Isolation Forest Detection')
    plt.legend(title='Prediction', labels=['Normal', 'Outlier'])
    
    plt.tight_layout()
    plt.savefig('viz_detection_comparison.png')
    plt.close()
    
    # 3. Visualization Before vs After Handling (Removal)
    # Show distribution of a key feature (e.g., NILAI_KLR)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['NILAI_KLR'])
    plt.title('Distribution Before Removal (Original)')
    plt.ylabel('NILAI_KLR (Scaled)')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_clean['NILAI_KLR'])
    plt.title('Distribution After Removal (Cleaned)')
    plt.ylabel('NILAI_KLR (Scaled)')
    
    plt.tight_layout()
    plt.savefig('viz_before_after_removal.png')
    plt.close()
    
    print("Visualizations saved: viz_confusion_matrix.png, viz_detection_comparison.png, viz_before_after_removal.png")

if __name__ == "__main__":
    main()

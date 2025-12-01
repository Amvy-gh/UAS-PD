# Implementation Results: Isolation Forest for Stock Outlier Detection

I have implemented the outlier detection method as described in the journal, adapted for your sales dataset.

## Methodology

1.  **Data Pre-processing**:
    *   **Label Encoding**: Applied to `NAMA_PRODUK` and `UNIT`.
    *   **Scaling**: Applied Standard Scaler to `QTY_MSK`, `NILAI_MSK`, `QTY_KLR`, `NILAI_KLR`.
    *   **Ground Truth Generation**: I calculated the "Stock Available" for each transaction. A transaction is flagged as an **Outlier (Stock Error)** if `QTY_KLR > Stock Available`.

2.  **Feature Selection (RFE)**:
    *   Used `ExtraTreesClassifier` to select the most important features for detecting Stock Errors.
    *   **Selected Features**: `NILAI_MSK`, `NILAI_KLR`, `NAMA_PRODUK_encoded`, `UNIT_encoded`.

3.  **Isolation Forest**:
    *   Trained on the selected features.
    *   Contamination Rate: **8.35%** (based on the actual rate of Stock Errors in the data).

4.  **Evaluation**:
    *   Compared the Isolation Forest predictions against the Stock Logic Ground Truth.

## Results

| Metric | Value |
| :--- | :--- |
| **Total Transactions** | 474,468 |
| **Actual Outliers (Stock Errors)** | 39,601 (8.35%) |
| **Detected Outliers (iForest)** | 39,594 |
| **True Positives (Correctly Detected)** | 4,263 |
| **False Positives (False Alarms)** | 35,331 |
| **Precision** | 11% |
| **Recall** | 11% |

### Confusion Matrix
```
[[399536 (Normal Correct),  35331 (False Alarm)],
 [ 35338 (Missed Error),    4263 (Detected Error)]]
```

## Analysis

The Isolation Forest detected **11%** of the stock errors. The low performance is likely due to the nature of the features used:
*   **State vs. Point Anomalies**: The "Stock Error" is a *contextual* anomaly (it depends on the *current stock level*, which changes over time).
*   **Feature Limitation**: The Isolation Forest was given only the *current transaction details* (`QTY`, `NILAI`, `PRODUCT`). It did not have access to the "Available Stock" history. Without knowing the available stock, the model cannot easily distinguish a valid sale of 10 items from an invalid sale of 10 items (when only 5 are in stock), as "selling 10 items" looks like a normal transaction.

To improve detection, we would need to include **stateful features** (e.g., "Stock Level Before Transaction") in the input to the Isolation Forest, but this essentially gives the model the answer.

## Step 4: Cleaning Simulation
*   **Removed**: 39,594 instances identified as outliers by iForest.
*   **Remaining Stock Errors**: 35,338 errors still exist in the "cleaned" dataset.
*   **Valid Data Lost**: 35,331 valid transactions were removed.

This suggests that for *this specific type of logical error* (Stock < 0), a rule-based approach (the Ground Truth logic itself) is superior to unsupervised anomaly detection on raw transaction features.

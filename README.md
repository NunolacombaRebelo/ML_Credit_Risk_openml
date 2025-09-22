# ML_Credit_Risk_openml
# Credit Risk Analysis — OpenML *credit-g*

Notebook that explores credit-risk classification (good vs. bad) using the OpenML *credit-g* dataset.  
I compare **Logistic Regression, SVM, Decision Tree, Random Forest, and MLP (Neural Networks)**, preprocess with a `ColumnTransformer` (OHE for categoricals + scaling for numericals), and evaluate with **Accuracy, F1-score, Confusion Matrix**, and **ROC**. I also tune hyperparameters and adjust the **decision threshold** to balance FP/FN.

## Quick start
    # create environment (example with pip)
    pip install pandas numpy scikit-learn matplotlib seaborn openml jupyter

    # run notebook
    jupyter notebook

Open **ML_Model_Credit_Risk.ipynb** and run all cells.

## Results (test set)
| Model                  | Accuracy | F1    |
|------------------------|:-------:|:-----:|
| **Logistic Regression**| **0.795** | **0.788** |
| SVM                    | 0.735   | 0.661 |
| Decision Tree          | 0.710   | 0.658 |
| Random Forest          | 0.705   | 0.671 |
| Neural Networks (MLP)  | 0.710   | 0.595 |

**Conclusion.** Logistic Regression achieved the best overall performance and the most balanced FP/FN trade-off. Threshold calibration can align decisions with the organization’s risk tolerance.

## Repository contents
- `ML_Model_Credit_Risk.ipynb` — full workflow (EDA → modeling → tuning → evaluation)
- `README.md` — project description and instructions


## Data
Dataset: **OpenML credit-g**. The notebook loads the data programmatically; raw data files are not committed.

## License
MIT (or update to your preferred license).

# Heart Diseases Prediction

## Problem Statement
Build a model to predict the presence of heart disease from clinical measurements and test results. The notebook demonstrates the full mini-ML project workflow: EDA, cleaning, feature engineering, model training, evaluation, and interpretation.

## Dataset
- **Name:** Heart Disease dataset (heart.csv)
- **Source:** UCI / common heart-disease CSV (included as `heart.csv` in this folder)
- **Rows / Columns:** 302 rows, ~14 original columns (see `heart.csv`)

## Tools Used
- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
- joblib

## Workflow
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering (one-hot encoding, age decade binning, scaling)
5. Model Building (`LogisticRegression`)
6. Evaluation (confusion matrix, ROC, metrics)
7. Insights & Recommendations

## Results
- **Model:** Logistic Regression
- **Key metrics:** Accuracy = 0.836, Recall = 0.848, ROC-AUC = 0.903
- **Top Factors / Drivers:** `sex`, `ca` (number of major vessels), `cp` (chest pain variants), `exang` (exercise-induced angina), `thalach` (max heart rate)

## Files to Inspect
- `Heart_Diseases.ipynb` — executed notebook with all steps and outputs. Open and view the rendered notebook to see charts and inline results.
- `heart.csv` — original raw dataset used as input.
- `outputs/metrics.txt` — final evaluation metrics summary.
- `outputs/predictions.csv` — test-set predictions and probabilities.
- `outputs/coefficients.csv` — model coefficients (feature importance for logistic regression).
- `images/` — EDA and evaluation figures (confusion matrix, ROC curve, correlation heatmap, etc.).
- `requirements.txt` — Python dependencies for reproducing the environment.

## Screenshots
Add screenshots from the `images/` folder, for example:

![EDA chart](images/01_disease_rate_by_sex_and_cp.png)
![Model evaluation](images/confusion_matrix.png)

## How to Reproduce
1. Install dependencies:
```powershell
pip install -r requirements.txt
```
2. Open the notebook in Jupyter or VS Code and run all cells top-to-bottom:
```powershell
jupyter notebook Heart_Diseases.ipynb
```

If you prefer not to re-run feature engineering, you can inspect the saved feature files in `outputs/` (`X_features.csv`, `y_target.csv`, `heart_features.csv`).

## Future Improvements
- Try ensemble models (Random Forest, XGBoost) and compare performance.
- Address class imbalance with sampling or threshold tuning if needed.
- Package feature engineering into a reusable script and add automated tests.

## Author
[Dhruv Kumar] — www.linkedin.com/in/dhruv-kumar-880b76308

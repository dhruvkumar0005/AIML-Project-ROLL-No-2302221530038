# Heart Diseases Prediction

## Project Overview
This project predicts the presence of heart disease using clinical and diagnostic measurements. It includes a complete ML workflow with data cleaning, exploratory data analysis, feature engineering, logistic regression modeling, and evaluation.

## Repository Structure
- `Dataset/` — raw input data
  - `heart.csv`
- `Notebook/` — primary Jupyter notebook
  - `Heart_Diseases.ipynb`
- `Images/` — saved EDA and evaluation figures
- `outputs/` — saved datasets, model artifacts, metrics, and predictions
- `requirements.txt` — Python dependencies
- `app.py` — Streamlit deployment app
- `README.md` — project documentation

## Dataset
- **Name:** Heart Disease dataset
- **Path:** `Dataset/heart.csv`
- **Description:** Clinical measurements for heart disease prediction, including age, chest pain type, resting blood pressure, cholesterol, and target label.

## Key Components
- Data cleaning and placeholder imputation for missing `ca` and `thal` values
- Exploratory data analysis with multiple visualizations
- Feature engineering: one-hot encoding, age decade binning, and scaling
- Logistic Regression model training and evaluation
- Saved artifacts for reproducible analysis

## Output Files
- `outputs/heart_cleaned.csv` — cleaned dataset after preprocessing
- `outputs/heart_features.csv` — engineered feature dataset
- `outputs/X_features.csv` — feature matrix used for modeling
- `outputs/y_target.csv` — target labels for modeling
- `outputs/standard_scaler.joblib` — fitted scaler object
- `outputs/logistic_model.joblib` — trained logistic regression model
- `outputs/predictions.csv` — test-set predictions and probabilities
- `outputs/metrics.txt` — accuracy, precision, recall, F1, and ROC-AUC
- `outputs/coefficients.csv` — feature coefficients from the model

## Images
- `Images/01_disease_rate_by_sex_and_cp.png`
- `Images/02_thalach_distribution_by_target.png`
- `Images/03_correlation_heatmap.png`
- `Images/04_oldpeak_vs_target.png`
- `Images/confusion_matrix.png`
- `Images/roc_curve.png`

## Reproduce the Analysis
1. Install dependencies:
```powershell
pip install -r requirements.txt
```
2. Open the notebook from the `Notebook/` folder:
```powershell
jupyter notebook Notebook/Heart_Diseases.ipynb
```
3. Run all cells from top to bottom.

> Note: The notebook saves generated charts into the root `Images/` folder and model outputs into the root `outputs/` folder.

## Streamlit App
1. Run the app from the project root:
```powershell
streamlit run app.py
```
2. Enter patient information in the form.
3. Click **Predict** to see the model result and probability.

## Streamlit Deployment
- Push the repository to GitHub.
- Create a new app on [Streamlit Cloud](https://streamlit.io/cloud).
- Connect the app to this repository.
- Set the main file to `app.py`.
- Streamlit Cloud will install dependencies from `requirements.txt` and deploy automatically.

## Results Summary
- **Model:** Logistic Regression
- **Accuracy:** ~0.836
- **Recall:** ~0.848
- **ROC-AUC:** ~0.903

## Future Improvements
- Compare logistic regression with tree-based or ensemble methods
- Improve feature selection and do more hyperparameter tuning
- Add end-to-end pipeline scripts for automation

## Author
Dhruv Kumar


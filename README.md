# Heart Diseases Prediction

## Project Overview
This project predicts the presence of heart disease using clinical and diagnostic measurements. It includes data cleaning, exploratory data analysis, feature engineering, logistic regression modeling, and a Streamlit user interface for prediction.

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
- **Description:** Clinical measurements used to predict heart disease, including age, chest pain type, blood pressure, cholesterol, heart rate, and other diagnostic features.

## Key Components
- Data cleaning and imputation for missing `ca` and `thal` values
- Exploratory data analysis with visualizations
- Feature engineering: one-hot encoding, age decade binning, and scaling
- Logistic Regression model training and evaluation
- Streamlit app for interactive prediction

## Output Files
- `outputs/heart_cleaned.csv` — cleaned dataset after preprocessing
- `outputs/heart_features.csv` — engineered feature dataset
- `outputs/X_features.csv` — feature matrix used for modeling
- `outputs/y_target.csv` — target labels for modeling
- `outputs/standard_scaler.joblib` — fitted scaler object
- `outputs/logistic_model.joblib` — trained logistic regression model
- `outputs/predictions.csv` — test-set predictions and probabilities
- `outputs/metrics.txt` — accuracy, precision, recall, F1, and ROC-AUC
- `outputs/coefficients.csv` — model coefficients for each feature

## Images
- `Images/01_disease_rate_by_sex_and_cp.png`
- `Images/02_thalach_distribution_by_target.png`
- `Images/03_correlation_heatmap.png`
- `Images/04_oldpeak_vs_target.png`
- `Images/confusion_matrix.png`
- `Images/roc_curve.png`

## How to Run Locally
1. Install dependencies:
```powershell
pip install -r requirements.txt
```
2. Run the Streamlit app:
```powershell
streamlit run app.py
```
3. Use the form to enter patient details and click **Predict**.

## Streamlit App Notes
- The app uses user-friendly dropdown labels for categorical inputs.
- It loads the saved model and scaler from `outputs/`.
- Predictions are shown with probability and risk interpretation.

## Streamlit Deployment
To deploy on Streamlit Cloud:
- Push this repository to GitHub.
- Create a new app on [Streamlit Cloud](https://streamlit.io/cloud).
- Connect the app to this repository.
- Set the main file to `app.py`.
- Streamlit Cloud installs dependencies from `requirements.txt` automatically.

## Results Summary
- **Model:** Logistic Regression
- **Accuracy:** ~0.836
- **Recall:** ~0.848
- **ROC-AUC:** ~0.903

## Future Improvements
- Compare logistic regression with tree-based or ensemble methods
- Improve feature selection and perform hyperparameter tuning
- Add a production-ready model pipeline and data validation

## Author
Dhruv Kumar


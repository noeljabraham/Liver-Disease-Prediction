# Liver Disease Prediction

A simple machine learning web app that predicts whether a patient is likely to have liver disease based on clinical input values.

This project uses:
- `scikit-learn` for model training (`RandomForestClassifier`)
- `StandardScaler` for feature scaling
- `Streamlit` for the web interface

## Project Structure

```text
.
|-- app.py                      # Streamlit web app
|-- train_model.py              # Training script for model + scaler
|-- liver_patient_dataset.csv   # Dataset used for training
|-- model.pkl                   # Trained model (generated)
|-- scaler.pkl                  # Trained scaler (generated)
`-- requirements.txt            # Python dependencies
```

## Features

- Interactive form for entering patient values
- Binary prediction:
  - High chance of liver disease
  - Low chance of liver disease
- Prediction confidence output
- Separate training pipeline to regenerate model artifacts

## Inputs Used

The model expects the following features in this order:
1. Age
2. Gender (`Male=1`, `Female=0`)
3. Total Bilirubin (TB)
4. Direct Bilirubin (DB)
5. Alkaline Phosphotase
6. SGPT
7. SGOT
8. Total Proteins (TP)
9. Albumin (ALB)
10. Albumin/Globulin Ratio

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Liver-Disease-Prediction.git
cd Liver-Disease-Prediction
```

### 2. Create and activate a virtual environment (recommended)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

## Retrain the Model

If you update the dataset or want fresh model files:

```bash
python train_model.py
```

This regenerates:
- `model.pkl`
- `scaler.pkl`

## Notes

- This project is for educational/demo purposes and is **not** a substitute for medical diagnosis.
- Ensure `model.pkl` and `scaler.pkl` exist before launching the app.

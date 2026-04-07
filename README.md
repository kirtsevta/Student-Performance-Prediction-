# Student Performance Prediction Dashboard

An end-to-end Streamlit application that predicts student exam scores from academic, lifestyle, and demographic factors. The app supports both single-student predictions and institution-scale batch predictions with downloadable outputs and visual analytics.

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Usage Guide](#usage-guide)
- [Input Data Format](#input-data-format)
- [Outputs](#outputs)
- [Model and Files](#model-and-files)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## Project Overview

This project is built to estimate a student's exam score (0-100 range) using machine learning. It is useful for:

- early academic risk identification
- performance planning and intervention
- quick what-if analysis for factors such as attendance, study time, and lifestyle indicators

The interface is designed for non-technical users such as educators, counselors, and academic coordinators.

## Problem Statement

Student performance is influenced by many interconnected factors (study habits, attendance, family context, sleep, prior scores, etc.). Manual interpretation is slow and subjective, especially when handling many students.

Organizations need:

- a consistent prediction method
- fast evaluation for one or many students
- actionable visual summaries for decision-making

## Solution Approach

The solution uses a supervised regression pipeline (Random Forest model in the current implementation) to learn relationships between student attributes and final exam score.

The app provides two workflows:

1. **Single Prediction:** Enter one student's details through a form and get an instant score prediction.
2. **Batch Prediction:** Upload CSV/Excel data for many students, validate schema, predict all rows, and export results.

## Key Features

- **Interactive Streamlit UI** with custom dashboard styling
- **Single-student prediction form** with rich input controls
- **Batch prediction** from both `.csv` and Excel files (`.xlsx`, `.xls`)
- **Input validation** before running batch inference
- **Visual analytics** for predicted score distribution and category insights
- **Feature importance chart** to show model influence factors
- **Download options** for prediction outputs (CSV/Excel)
- **Dark mode and improved UX layout** for professional presentation

## Tech Stack

- **Language:** Python 3.x
- **Frontend / App Layer:** Streamlit
- **Data Processing:** pandas, NumPy
- **Machine Learning:** scikit-learn (`RandomForestRegressor`)
- **Visualization:** Matplotlib
- **Model Serialization:** pickle

## Project Structure

```text
case study/
|- app.py                            # Main Streamlit application (UI + prediction workflows)
|- requirements.txt                  # Python dependencies
|- student_model.pkl                 # Trained regression model artifact
|- StudentPerformanceFactors.csv     # Source/reference dataset
|- README.md                         # Project documentation
```

## How It Works

1. The app starts and loads serialized artifacts (model, encoders, feature schema).
2. User input is collected from:
   - form fields (single mode), or
   - uploaded file (batch mode).
3. Categorical values are encoded to numeric format using stored label encoders.
4. Features are aligned to model training columns.
5. The model predicts exam scores.
6. Predicted scores are clipped to valid score bounds and displayed/exported.
7. Analytics charts and summary metrics are generated for interpretation.

## Getting Started

### 1) Clone or download this project

Place all project files in one directory and open that directory in your terminal.

### 2) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4) Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Usage Guide

### Single Prediction

1. Open the Dashboard page from sidebar navigation.
2. Fill all required fields in the student form.
3. Submit to generate a predicted exam score.
4. Review score, confidence context (if shown), and feature-importance visualization.
5. Download single prediction summary if needed.

### Batch Prediction

1. Switch to the Batch Prediction tab.
2. Upload a valid CSV/Excel file containing required columns.
3. Run schema validation.
4. Start prediction for all records.
5. Download outputs in CSV/Excel formats.
6. Review analytics plots for institution-level patterns.

## Input Data Format

Batch files must:

- include all required feature columns expected by the model
- keep compatible data types (categorical vs numeric)
- avoid missing mandatory values

Tip: Use the provided dataset (`StudentPerformanceFactors.csv`) as a reference template for field naming and structure.

## Outputs

The application can produce:

- predicted exam score per student
- categorized score insights (if enabled in workflow)
- dashboard charts for aggregate analysis
- downloadable prediction reports:
  - `batch_predictions.csv`
  - `batch_predictions.xlsx`
  - single-student summary export

## Model and Files

Required model artifacts should be present in the project root:

- `student_model.pkl`
- `label_encoders.pkl` (if your inference flow uses encoded categorical mappings)
- `feature_columns.pkl` (for exact training-schema alignment)

If encoder/schema artifacts are missing, behavior depends on fallback logic implemented in `app.py`.

## Troubleshooting

- **App not starting:** confirm dependencies are installed from `requirements.txt`.
- **Model load error:** verify `.pkl` artifacts exist in the same folder as `app.py`.
- **Batch validation fails:** check uploaded file columns and datatypes.
- **Prediction errors:** ensure no unexpected null/blank values in required fields.
- **Port issue:** run with custom port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

## Future Enhancements

- model versioning and experiment tracking panel
- explainability module (SHAP / advanced feature contribution views)
- REST API layer for external integrations
- authentication and role-based dashboard access
- deployment recipes (Streamlit Community Cloud, Docker, cloud VM)

## License

Add a license file (for example, MIT) if you plan to distribute this project publicly.

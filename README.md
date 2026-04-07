# Student Performance Prediction (Streamlit)

A professional Streamlit web app to predict student exam performance using machine learning.

## Features

- Single student prediction form
- Batch prediction from CSV/Excel upload
- Analytics visuals and score bands
- Styled dashboard UI with dark mode toggle

## Tech Stack

- Python
- Streamlit
- pandas, numpy
- scikit-learn
- matplotlib

## Project Structure

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `student_model.pkl` - Trained model artifact
- `StudentPerformanceFactors.csv` - Sample/reference dataset

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

## Notes

- Keep `student_model.pkl` in the project root for model loading.
- If model encoder files are missing, app may run in demo/synthetic behavior depending on logic in `app.py`.

## Future Improvements

- Add model/version metadata panel
- Add export-ready PDF report
- Add API endpoint layer for integration

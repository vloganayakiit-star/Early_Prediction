# Early At-Risk Student Prediction — Streamlit Prototype

A runnable prototype based on the uploaded paper, **“A Resource-Efficient Deep Learning Model for Early At-Risk Student Prediction: Balancing Accuracy and Deployability for Low-Infrastructure Institutions.”**

## Important research limitation

The included demonstration dataset is synthetic. Its metrics demonstrate that the code pipeline runs; they are **not empirical evidence about real students**. The paper describes a public student outcome dataset augmented with simulated weekly attendance/LMS features. Any final paper or presentation must disclose this and must not describe simulated records as actual institutional observations.

## Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- 8 GB RAM should be sufficient for the small model
- Internet access for initial package installation

## Install

Open Command Prompt in this folder:

```bat
python --version
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If TensorFlow installation fails, check that you are using a supported Python version (Python 3.10/3.11 is recommended for this package set).

## Create demonstration data

```bat
python data_simulator.py
```

This writes `data\synthetic_demo_data.csv`. The rows are synthetic and must not be represented as real student records.

## Train and evaluate

```bat
python train_model.py
```

Optional: use a labelled CSV matching the required schema:

```bat
python train_model.py --csv data\your_training_data.csv --epochs 15
```

The script saves the LSTM model, scaler, metadata, model comparison table, training history, and test predictions under `models\` and `results\`.

## Run the dashboard

```bat
python -m streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. Open that address in your browser.

## CSV schema

The training file must contain `At_Risk` with values `0` (Safe) and `1` (At-Risk), plus consistent sequential feature columns such as:

- `week1_attendance_pct`, `week1_lms_logins`, `week1_resource_accesses`, `week1_submission_delay_days`, `week1_assessment_score`
- matching `week2_...` through `week6_...` columns

Use the same feature names in every week. Prediction files do not need `At_Risk`.

## Research integrity checklist

- Verify the public dataset source, version, dimensions, target, and class distribution.
- Exclude target and end-of-semester information from model inputs.
- Fit all preprocessing on the training partition only.
- Disclose every simulated feature and the simulation procedure.
- Report only measured performance and efficiency values.
- Do not claim real-world validity without validation on genuine institutional records.
- Do not claim public cloud deployment unless it has been successfully deployed and tested.

## Project files

- `app.py`: Streamlit user interface
- `train_model.py`: LSTM and Logistic Regression training/evaluation
- `model_utils.py`: input validation and sequence preparation
- `data_simulator.py`: synthetic demonstration data generator
- `requirements.txt`: Python dependencies
- `data/`, `models/`, `results/`, `figures/`: runtime artifacts

## Enhanced analytics and explainability

The dashboard includes KPI cards, searchable/filterable prediction results, weekly trends, numeric distributions, correlation heatmap, CSV prediction export, PDF summary export, and an **Evaluation & Explainability** tab. After training, that tab displays a confusion matrix, ROC curve, LSTM-vs-Logistic-Regression comparison, measured CPU timing/model file size, and global test-set permutation feature importance.

The explainability output is a global feature-importance estimate: it measures the change in test ROC-AUC when one feature is permuted. It is not causal evidence and is not a guaranteed individual-student explanation. Evaluation plots and feature importance are generated only after a successful training run.

## What has been checked

- Python syntax compilation completed for `app.py`, `train_model.py`, `model_utils.py`, and `data_simulator.py`.
- The included CSV was parsed successfully into 1,200 records, 6 weeks, and 5 features per week; it contains both target classes.
- The live Streamlit dashboard and TensorFlow training were **not** run in the build environment because Streamlit and TensorFlow are not installed there. Install `requirements.txt` on your Windows machine and run training to populate evaluation figures and measured efficiency values.

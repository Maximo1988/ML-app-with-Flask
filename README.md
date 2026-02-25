# Drug Recommendation App (Flask + Random Forest)

This project aims to predict the most suitable medication type for a patient based on health-related features (`Age`, `Sex`, `BP`, `Cholesterol`, `Na_to_K`). The workflow includes EDA in a notebook, model training with Random Forest, and an interactive Flask app where users enter patient data and receive a predicted drug.

## Project Organization

- `src/app.py`: Flask app, model training pipeline, and prediction endpoint.
- `src/templates/index.html`: User interface with patient form and prediction output.
- `src/explore.ipynb`: EDA and model evaluation notebook.
- `dataset/drug.csv`: Main dataset used by notebook and app.
- `dataset/drug_data.py`: Utility script to generate/update `dataset/drug.csv` from the original source path.
- `requirements.txt`: Minimal required dependencies.
- `data/`: Optional folders (`raw`, `interim`, `processed`) for future dataset versioning.
- `models/`: Optional folder for saved models/artifacts.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. If needed, regenerate local dataset file:

```bash
python dataset/drug_data.py
```

## Run the Web App

Start Flask (served with Waitress):

```bash
python src/app.py
```

Then open:

`http://localhost:5000`

## EDA and Model Validation

Open notebook:

- `src/explore.ipynb`

Main sections included:

- Data loading and structure checks (`head`, `shape`, `info`, `describe`)
- Missing values check
- EDA visualizations (target distribution, categorical vs target, numeric distributions, boxplots, correlation)
- Random Forest evaluation (accuracy, classification report, confusion matrix)

## Current Tech Stack

- Flask
- Waitress
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
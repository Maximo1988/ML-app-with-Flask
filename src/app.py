from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from waitress import serve

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DEFAULT_DATASET_PATH = DATASET_DIR / "drug.csv"


def load_dataset(data_path: Path):
	return pd.read_csv(data_path)


def train_model(dataset: pd.DataFrame):
	feature_columns = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
	target_column = "Drug"

	X = dataset[feature_columns]
	y = dataset[target_column]

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	preprocess = ColumnTransformer(
		transformers=[
			("categorical", OneHotEncoder(handle_unknown="ignore"), ["Sex", "BP", "Cholesterol"]),
			("numeric", "passthrough", ["Age", "Na_to_K"]),
		]
	)

	model = Pipeline(
		steps=[
			("preprocess", preprocess),
			("classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
		]
	)

	model.fit(X_train, y_train)
	predictions = model.predict(X_test)
	accuracy = accuracy_score(y_test, predictions)

	return model, accuracy


dataset = load_dataset(DEFAULT_DATASET_PATH)
model, model_accuracy = train_model(dataset)


@app.route("/", methods=["GET", "POST"])
def index():
	predicted_drug = None
	prediction_confidence = None
	form_error = None
	form_data = {
		"Age": "",
		"Sex": "M",
		"BP": "HIGH",
		"Cholesterol": "HIGH",
		"Na_to_K": "",
	}

	if request.method == "POST":
		form_data = {
			"Age": request.form.get("Age", "").strip(),
			"Sex": request.form.get("Sex", "M"),
			"BP": request.form.get("BP", "HIGH"),
			"Cholesterol": request.form.get("Cholesterol", "HIGH"),
			"Na_to_K": request.form.get("Na_to_K", "").strip(),
		}

		try:
			input_data = pd.DataFrame(
				[
					{
						"Age": int(form_data["Age"]),
						"Sex": form_data["Sex"],
						"BP": form_data["BP"],
						"Cholesterol": form_data["Cholesterol"],
						"Na_to_K": float(form_data["Na_to_K"]),
					}
				]
			)

			predicted_drug = model.predict(input_data)[0]
			probabilities = model.predict_proba(input_data)[0]
			prediction_confidence = round(float(probabilities.max()) * 100, 2)
		except ValueError:
			form_error = "Verifica los campos numéricos: Edad y Na_to_K deben ser válidos."

	dataset_info = {
		"filename": str(DEFAULT_DATASET_PATH.relative_to(BASE_DIR)),
		"rows": int(dataset.shape[0]),
		"columns": int(dataset.shape[1]),
		"table_html": dataset.head(10).to_html(classes="table", index=False),
	}
	return render_template(
		"index.html",
		dataset_info=dataset_info,
		model_accuracy=round(model_accuracy, 4),
		predicted_drug=predicted_drug,
		prediction_confidence=prediction_confidence,
		form_error=form_error,
		form_data=form_data,
	)


if __name__ == "__main__":
	serve(app, host="0.0.0.0", port=5000)

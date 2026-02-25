# App de Recomendación de Medicamento (Flask + Random Forest)

Este proyecto busca predecir qué tipo de medicamento es más adecuado para un paciente según sus características de salud (`Age`, `Sex`, `BP`, `Cholesterol`, `Na_to_K`). El flujo incluye EDA en notebook, entrenamiento del modelo Random Forest y una app Flask donde el usuario ingresa datos y recibe una predicción.

## Organización del Proyecto

- `src/app.py`: App Flask, pipeline de entrenamiento y endpoint de predicción.
- `src/templates/index.html`: Interfaz web con formulario de paciente y resultado.
- `src/explore.ipynb`: Notebook para EDA y evaluación del modelo.
- `dataset/drug.csv`: Dataset principal (de Kaggle) usado por notebook y app.
- `dataset/drug_data.py`: Script auxiliar para generar/actualizar `dataset/drug.csv` desde la ruta de origen.
- `requirements.txt`: Dependencias mínimas necesarias.
- `data/`: Carpetas opcionales (`raw`, `interim`, `processed`) para futuras versiones del dataset.
- `models/`: Carpeta opcional para guardar modelos/artefactos.

## Instalación

1. Instala dependencias:

```bash
pip install -r requirements.txt
```

2. Si necesitas regenerar el CSV local:

```bash
python dataset/drug_data.py
```

## Ejecutar la Aplicación

Inicia la app (servida con Waitress):

```bash
python src/app.py
```

Luego abre en el navegador:

`http://localhost:5000`

## EDA y Validación del Modelo

Abre el notebook:

- `src/explore.ipynb`

Secciones incluidas:

- Carga y revisión inicial de datos (`head`, `shape`, `info`, `describe`)
- Revisión de valores faltantes
- Visualizaciones EDA (distribución objetivo, categóricas vs objetivo, numéricas, boxplots, correlación)
- Evaluación Random Forest (accuracy, classification report, matriz de confusión)

## Stack Actual

- Flask
- Waitress
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
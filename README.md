# Streaming Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.x-3670A0?style=flat-square&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-006ACC?style=flat-square)](https://xgboost.ai/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-FA0F00?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/streaming-subscription-churn-model/team)

Modelo de clasificación binaria para predecir **abandono de suscripción** en un servicio de streaming. Entrenado sobre **125,000 usuarios** con **20 features** comportamentales y de cuenta. El mejor modelo (Random Forest con tuning) alcanza **AUC-ROC 0.9347** y **F1 0.8524** en el conjunto de validación.

> Submission para la [competición de Kaggle "Streaming Subscription Churn Model"](https://www.kaggle.com/competitions/streaming-subscription-churn-model/team). Comparativa de tres familias de algoritmos (Random Forest, XGBoost, Logistic Regression), tuning por búsqueda aleatoria/grid, validación con StratifiedKFold y análisis de importancia de variables.

---

## Resultados clave

| Modelo | AUC-ROC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Random Forest** (tuned) | **0.9347** | **84.78%** | **84.89%** | **85.60%** | **85.24%** |
| Logistic Regression (tuned) | 0.8935 | 80.44% | 80.97% | 80.91% | 80.94% |
| XGBoost (tuned) | 0.8732 | 77.24% | 79.08% | 75.68% | 77.35% |

**AUC-ROC > 0.9** se considera excelente para problemas de clasificación binaria con clases moderadamente balanceadas. La diferencia de ~4 puntos entre RF y LogReg sugiere que las relaciones son **no lineales** y que el ensemble captura interacciones que un modelo lineal no.

## Insights de negocio

Aprendidos del análisis (no asumidos):

- **Tipo de suscripción dispara el churn**:
  - Free: **79.4%** de churn
  - Premium / Family: **34-35%** de churn
- **Menor uso semanal** correlaciona fuertemente con abandono.
- **Más consultas a soporte** = más probabilidad de cancelar (consultas como señal anticipada de fricción).
- **Pausas previas en la suscripción** son predictoras tempranas.

### Top 5 features por importancia (Random Forest)

1. `weekly_hours` — horas semanales de uso
2. `customer_service_inquiries` — consultas a soporte
3. `subscription_type` — tipo de plan
4. `song_skip_rate` — tasa de skip de canciones
5. `num_subscription_pauses` — pausas de suscripción

---

## Stack

| Categoría | Tecnologías |
|---|---|
| ML | scikit-learn, XGBoost |
| Data | pandas, numpy |
| Viz | matplotlib, seaborn |
| Notebook | Jupyter |
| Persistencia | pickle (sklearn) |

## Dataset

| | |
|---|---|
| Fuente | [Kaggle Competition](https://www.kaggle.com/competitions/streaming-subscription-churn-model/team) |
| Train | 125,000 usuarios |
| Test | conjunto separado (`test.csv`) |
| Features | 20 (mezcla numérica + categórica) |
| Target | `churn` (binario) |

### Variables

**Numéricas (7)**
- `age` — edad del usuario
- `weekly_hours` — horas semanales de uso
- `average_session_length` — duración media de sesión
- `song_skip_rate` — tasa de skip
- `weekly_songs_played` — canciones por semana
- `num_subscription_pauses` — pausas de suscripción
- `customer_tenure_years` — antigüedad como cliente

**Categóricas (5)**
- `subscription_type` — Free / Premium / Family / Student
- `payment_plan` — Monthly / Yearly
- `payment_method`
- `location`
- `customer_service_inquiries`

---

## Metodología

### 1. Preparación
- Encoding de categóricas: dos versiones del dataset preparadas
  - **Label encoding** para árboles (Random Forest, XGBoost)
  - **One-hot encoding** para Logistic Regression
- Train/validation split: 80/20 estratificado por target (`stratify=y`, `random_state=42`).

### 2. Validación cruzada
**StratifiedKFold** para todos los modelos. Esto mantiene la proporción de clases en cada fold — clave cuando hay un desbalance moderado en `churn`.

### 3. Tuning de hiperparámetros
- **Random Forest** y **XGBoost** → `RandomizedSearchCV` (eficiente para espacios de búsqueda grandes).
- **Logistic Regression** → `GridSearchCV` (espacio más pequeño, búsqueda exhaustiva).
- Métrica de optimización: **ROC-AUC**.

### 4. Evaluación
Sobre el set de validación se computan:
- ROC-AUC, accuracy, precision, recall, F1.
- Confusion matrix.
- Curvas ROC y Precision-Recall.
- Feature importance (RF, XGB) y coeficientes (LogReg).

### Por qué Random Forest gana aquí

- **Captura no linealidades** sin necesidad de feature engineering manual.
- **Robusto a features mixtas** (numéricas + categóricas codificadas).
- **Bajo riesgo de overfitting** con CV + tuning, frente a XGBoost que es más sensible al ajuste fino.
- En este dataset específico, XGBoost se quedó atrás — probablemente requiere más tuning agresivo (`max_depth`, `learning_rate`, `subsample`) que no compensó frente al RF base.

---

## Reproducir

### Setup

```bash
git clone https://github.com/jbeleno/streaming-churn-prediction-model.git
cd streaming-churn-prediction-model

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

> **macOS:** XGBoost necesita OpenMP. `brew install libomp` si todavía no lo tienes.

### Entrenamiento (CLI)

```bash
# Entrenamiento completo (n_iter=20, ~10-20 min según hardware)
python -m src.train

# Smoke test rápido con subsample de 5k filas y n_iter=3 (~1 min)
python -m src.train --quick --n-iter 3

# Entrenar solo un subconjunto de modelos
python -m src.train --models rf logreg

# Custom output directory
python -m src.train --output ./my_artifacts
```

Genera en `artifacts/`:

- `rf.joblib` — Random Forest (Pipeline completo: preprocessor + estimator)
- `xgboost.joblib` — XGBoost (idem)
- `logreg.joblib` — Logistic Regression (idem)
- `metrics.json` — métricas de validación de cada modelo

### Inferencia (CLI)

```bash
python -m src.predict --model artifacts/rf.joblib --input test.csv --output predictions.csv
```

Salida: CSV con `prediction` (0/1) y `churn_probability` (P(churn=1)).

### Inferencia (Python)

```python
import joblib
import pandas as pd
from src.preprocessing import add_tenure_features

pipeline = joblib.load("artifacts/rf.joblib")

X_new = pd.read_csv("nuevos_usuarios.csv")
X_new = add_tenure_features(X_new)        # genera customer_tenure_*
preds = pipeline.predict(X_new)
proba = pipeline.predict_proba(X_new)[:, 1]
```

El `Pipeline` ya contiene el preprocessor (StandardScaler + OneHotEncoder), así que **no hay que aplicar transformaciones manualmente**. Esto es la diferencia clave con la versión 1.x basada en `pickle` + transformaciones inline en el notebook.

### Tests unitarios

```bash
pytest tests/ -v
# 7 tests cubriendo preprocessing, encoding de unknowns, y pipeline end-to-end
```

### Notebook original

`modelochurd.ipynb` se mantiene como referencia histórica del análisis exploratorio (EDA + iteración de modelos). Para reproducir resultados, **usar el CLI**.

---

## Estructura del proyecto

```
streaming-churn-prediction-model/
├── src/                       # Pipeline modular (v2.0)
│   ├── config.py              # Constantes, columnas, hiperparámetros, paths
│   ├── preprocessing.py       # ColumnTransformer + Pipeline + feature engineering
│   ├── train.py               # CLI de entrenamiento (RandomForest, XGBoost, LogReg)
│   ├── predict.py             # CLI de inferencia
│   └── eval.py                # Métricas con dataclass ModelMetrics
├── tests/
│   └── test_preprocessing.py  # 7 tests sobre el pipeline
├── modelochurd.ipynb          # Notebook EDA original (referencia histórica)
├── train.csv                  # 125k filas
├── test.csv
├── requirements.txt           # Runtime deps (pandas, sklearn, xgboost, joblib)
├── requirements-dev.txt       # + pytest
├── pyproject.toml             # Versioning, pytest config
├── LICENSE                    # MIT
└── README.md
```

---

## Mejoras pendientes (deuda técnica reconocida)

- **SMOTE / class weighting**: si hay desbalance significativo, probar oversampling de la clase minoritaria.
- **Tuning más agresivo de XGBoost**: el resultado actual (AUC 0.873) probablemente sube con búsqueda sobre `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`.
- **SHAP** para interpretabilidad por instancia, no solo feature importance global.
- **CalibratedClassifierCV** si las probabilidades se van a usar para decisiones de negocio (la calibración importa más que el AUC para retención dirigida).
- **CI con GitHub Actions** corriendo `pytest` en cada PR.
- **MLflow tracking** para registrar runs, hiperparámetros y artefactos automáticamente.
- **Validación temporal**: si los datos tienen orden temporal, usar `TimeSeriesSplit` en lugar de `StratifiedKFold` para evitar leak.

### Modernización aplicada en v2.0

Lo que **ya está hecho** (vs v1.x notebook-only):

- ✅ Notebook → módulos en `src/` (config, preprocessing, train, predict, eval).
- ✅ `Pipeline` scikit-learn unificado con `ColumnTransformer` (StandardScaler + OneHotEncoder).
- ✅ `pickle` → `joblib` (estándar sklearn, eficiente con arrays NumPy).
- ✅ CLIs con `argparse` (`--quick`, `--n-iter`, `--models`, `--output`).
- ✅ Tests unitarios con pytest (7 tests sobre preprocessing).
- ✅ Persistencia de `Pipeline` completo (preprocessor + modelo) en un solo `.joblib` — sin transformar manualmente al inferir.
- ✅ Manejo de categorías nunca vistas (`OneHotEncoder(handle_unknown='ignore')`).
- ✅ `pyproject.toml` con versioning + pytest config.

---

## Visualizaciones incluidas en el notebook

- Distribución de la variable target.
- Matrices de correlación.
- Feature importance (RF, XGB) y coeficientes (LogReg).
- Confusion matrix por modelo.
- Curvas ROC y Precision-Recall.
- Análisis de churn por segmento (subscription_type, payment_plan, etc.).

---

## Licencia

[MIT](./LICENSE).

## Autor

**Jesús Beleño**
- LinkedIn: [jesus-beleno](https://www.linkedin.com/in/jesus-beleno/)
- GitHub: [@jbeleno](https://github.com/jbeleno)

## Reconocimientos

- Dataset: [Kaggle — Streaming Subscription Churn Model](https://www.kaggle.com/competitions/streaming-subscription-churn-model/team)
- Stack: scikit-learn, XGBoost, pandas, matplotlib, seaborn.

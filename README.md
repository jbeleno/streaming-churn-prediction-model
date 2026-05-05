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

```bash
git clone https://github.com/jbeleno/streaming-churn-prediction-model.git
cd streaming-churn-prediction-model

pip install pandas numpy scikit-learn xgboost seaborn matplotlib

jupyter notebook modelochurd.ipynb
# Ejecutar todas las celdas en orden
```

Al ejecutar el notebook completo se generan los modelos serializados:

- `best_rf_label.pkl` — Random Forest (mejor modelo)
- `best_xgb_label.pkl` — XGBoost
- `best_logistic_regression.pkl` — Logistic Regression

> No se incluyen en el repositorio por tamaño/peso de archivos.

## Inferencia con el modelo entrenado

```python
import pickle
import pandas as pd

with open("best_rf_label.pkl", "rb") as f:
    model = pickle.load(f)

# X_new debe tener las mismas 20 columnas que X_train_label,
# con el mismo Label Encoding aplicado a las categóricas.
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]  # P(churn=1)
```

> Para usar en producción, considerar migrar de `pickle` a `joblib` (estándar sklearn, más eficiente para modelos grandes con arrays NumPy) o exportar a ONNX para portabilidad multi-runtime.

---

## Estructura del proyecto

```
streaming-churn-prediction-model/
├── modelochurd.ipynb          # Notebook principal (EDA + tuning + evaluación)
├── train.csv                  # Dataset de entrenamiento (125k filas)
├── test.csv                   # Dataset de prueba
├── LICENSE                    # MIT
└── README.md
```

---

## Mejoras pendientes (deuda técnica reconocida)

- **Pipeline scikit-learn unificado** (`Pipeline`) que combine encoding + escalado + modelo en un solo objeto. Hoy las transformaciones están sueltas en el notebook.
- **SMOTE / class weighting**: si hay desbalance significativo, probar oversampling de la clase minoritaria.
- **Tuning más agresivo de XGBoost**: el resultado actual (AUC 0.873) probablemente sube con búsqueda sobre `max_depth`, `subsample`, `colsample_bytree`, `learning_rate`.
- **SHAP** para interpretabilidad por instancia, no solo feature importance global.
- **CalibratedClassifierCV** si las probabilidades se van a usar para decisiones de negocio (la calibración importa más que el AUC para retención dirigida).
- **Migrar pickle → joblib** para serialización.
- **Tests unitarios** sobre las funciones de preprocesamiento.
- **Refactor del notebook a módulos** (`src/preprocessing.py`, `src/models.py`, `src/eval.py`) para reproducibilidad y CI.

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

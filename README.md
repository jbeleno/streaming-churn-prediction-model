# 🎵 Streaming Churn Prediction Model

## 📊 Project Description

This project develops a Machine Learning model to predict user churn in streaming services. The model uses advanced classification techniques to identify behavioral patterns that indicate subscription cancellation risk.

## 🎯 Objective

Identify the probability that customers will abandon the streaming service, enabling proactive retention strategies and optimizing marketing resources.

## 📈 Key Results

- **AUC-ROC: 93.47%** (Excellent predictive capability)
- **Accuracy: 84.78%** (High overall precision)
- **Best Model: Random Forest**
- **Dataset: 125,000 users** with 20 features

## 🏗️ Project Architecture

```
streaming-churn-prediction-model/
├── modelochurd.ipynb          # Main notebook with complete analysis
├── train.csv                  # Training dataset
├── test.csv                   # Test dataset
└── README.md                  # This file
```

**Nota:** Los modelos entrenados (`best_rf_label.pkl`, `best_xgb_label.pkl`, `best_logistic_regression.pkl`) se generan automáticamente al ejecutar el notebook completo.

## 🔍 Data Analysis

### Key Variables Identified:
1. **`weekly_hours`** - Weekly usage hours (most important)
2. **`customer_service_inquiries`** - Customer service inquiries
3. **`subscription_type`** - Subscription type
4. **`song_skip_rate`** - Song skip rate
5. **`num_subscription_pauses`** - Subscription pauses

### Business Insights:
- **Free users:** 79.4% churn rate
- **Premium/Family users:** 34-35% churn rate
- **Lower weekly usage** = Higher churn risk
- **More support inquiries** = Higher abandonment probability

## 🤖 Implemented Models

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| **Random Forest** | **0.9347** | **84.78%** | **84.89%** | **85.60%** | **85.24%** |
| Logistic Regression | 0.8935 | 80.44% | 80.97% | 80.91% | 80.94% |
| XGBoost | 0.8732 | 77.24% | 79.08% | 75.68% | 77.35% |

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine Learning
- **XGBoost** - Boosting algorithm
- **Seaborn/Matplotlib** - Visualizations
- **NumPy** - Numerical computation

## 📋 Dataset Features

### Numerical Variables:
- `age` - User age
- `weekly_hours` - Weekly usage hours
- `average_session_length` - Average session duration
- `song_skip_rate` - Song skip rate
- `weekly_songs_played` - Songs played per week
- `num_subscription_pauses` - Number of subscription pauses
- `customer_tenure_years` - Customer tenure

### Categorical Variables:
- `subscription_type` - Subscription type (Free, Premium, Family, Student)
- `payment_plan` - Payment plan (Monthly, Yearly)
- `payment_method` - Payment method
- `location` - User location
- `customer_service_inquiries` - Customer service inquiry frequency

## 🚀 Installation and Usage

### Prerequisites:
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
```

### Execution:
1. Clone the repository
2. Open `modelochurd.ipynb` in Jupyter Notebook
3. Run all cells to reproduce the complete analysis
4. Los modelos entrenados se guardarán automáticamente como archivos `.pkl`

### Using Trained Models:
```python
import pickle

# Load model (se genera al ejecutar el notebook)
with open('best_rf_label.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_new_data)
probabilities = model.predict_proba(X_new_data)
```

**Importante:** Los archivos `.pkl` de los modelos se crean automáticamente al ejecutar todas las celdas del notebook `modelochurd.ipynb`. Si no existen, ejecuta el notebook completo para generarlos.

## 📊 Evaluation Metrics

### Primary Metric: AUC-ROC
- **0.5:** Random performance
- **0.7-0.8:** Good
- **0.8-0.9:** Very good
- **0.9+:** Excellent
- **1.0:** Perfect

### Complementary Metrics:
- **Accuracy:** Proportion of correct predictions
- **Precision:** Efficiency of positive predictions
- **Recall:** Ability to capture real cases
- **F1-Score:** Balance between precision and recall

## 💡 Business Applications

### Retention Strategies:
1. **User segmentation** by churn risk
2. **Personalized campaigns** for high-risk users
3. **Marketing resource optimization**
4. **User experience improvement**

### KPIs to Monitor:
- Churn rate by segment
- Retention campaign effectiveness
- Loyalty strategy ROI
- Customer satisfaction

## 🔬 Methodology

1. **Exploratory Data Analysis (EDA)**
   - Variable distribution
   - Correlations
   - Missing value analysis

2. **Data Preparation**
   - Categorical variable encoding
   - Numerical variable scaling
   - Feature engineering

3. **Modeling**
   - Multiple algorithm training
   - Hyperparameter optimization
   - Cross-validation

4. **Evaluation**
   - Model comparison
   - Feature importance analysis
   - Result interpretation

## 📈 Included Visualizations

- Target variable distribution
- Correlation matrices
- Feature importance
- Confusion matrices
- ROC and Precision-Recall curves
- Churn analysis by categories

## 📄 License

This project is under the MIT License. See the `LICENSE` file for more details.

## 👨‍💻 Author

**Your Name**
- LinkedIn: https://www.linkedin.com/in/jesus-beleno/
- Email: jesusbelenov@gmail.com

## 🙏 Acknowledgments

- Dataset provided by [https://www.kaggle.com/competitions/streaming-subscription-churn-model/team]
- Data Science community
- Open source tools used

---

⭐ **If this project was helpful, please give it a star!** 
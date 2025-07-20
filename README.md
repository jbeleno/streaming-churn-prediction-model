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
├── best_rf_label.pkl          # Trained Random Forest model
├── best_xgb_label.pkl         # Trained XGBoost model
├── best_logistic_regression.pkl # Trained Logistic Regression model
└── README.md                  # This file
```

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

### Using Trained Models:
```python
import pickle

# Load model
with open('best_rf_label.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_new_data)
probabilities = model.predict_proba(X_new_data)
```

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

## 🤝 Contributing

Contributions are welcome. Please:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is under the MIT License. See the `LICENSE` file for more details.

## 👨‍💻 Author

**Your Name**
- LinkedIn: [Your LinkedIn]
- Email: your.email@example.com

## 🙏 Acknowledgments

- Dataset provided by [dataset source]
- Data Science community
- Open source tools used

## 📞 Contact

For questions or collaborations:
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [Your LinkedIn]
- 🐦 Twitter: [@your_twitter]

---

⭐ **If this project was helpful, please give it a star!** 
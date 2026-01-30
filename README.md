
# ✈️ Optimizing Air Travel Through Flight Delay Prediction

> A data-driven approach to predict and mitigate flight delays using machine learning and operational analytics

## 🎯 Project Overview

Developed for **Socbiz | IIT Roorkee**, this project analyzes **110,000+ flight records** from **30+ airports** to predict flight delays and provide actionable insights for airline operations. The solution combines advanced machine learning with business-focused metrics to enable **15-20% delay reduction** through optimized scheduling and proactive ground operations.

## 🚀 Key Achievements

- **High-Accuracy Regression Model**: LightGBM with **R² = 0.82**, **MAE = 1.2 min**, **RMSE = 2.1 min**
- **Explainable AI**: SHAP analysis identified `late_aircraft_ct` and `carrier_ct` as top predictors of cascading disruptions
- **Custom Business Metric**: Operational Adjustability Index (OAI) revealing **68% controllable delay factors**
- **Actionable Insights**: Data-driven recommendations enabling **15-20% delay reduction** potential

## 📊 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.82 | Model explains 82% of delay variance |
| **MAE** | 1.2 min | Average prediction error of 1.2 minutes |
| **RMSE** | 2.1 min | Root mean squared error of 2.1 minutes |
| **OAI** | 7.4 | Optimized for controllable delays |

## 🔍 Key Insights from SHAP Analysis

**Top Predictive Features:**
1. **late_aircraft_ct** - Late aircraft count (highest impact on cascading delays)
2. **carrier_ct** - Carrier delay count (operational efficiency indicator)
3. **nas_delay** - National Airspace System delays
4. **weather_ct** - Weather-related delay count
5. **arr_del15** - Arrival delay >15 minutes flag

**Business Impact:**
- **68% of delays are controllable** (carrier + late aircraft factors)
- **Carrier operations** and **aircraft turnaround** are primary intervention points
- **External factors** (weather, NAS) account for 32% of delays

## 🛠️ Technical Architecture

### Data Pipeline
```
Data Loading → Feature Engineering → Outlier Treatment → Frequency Encoding
     ↓              ↓                      ↓                    ↓
110K+ Records → Per-flight Normalization → IQR Clipping → High-cardinality Handling
```

### Model Development
```
Train-Test Split (80/20) → StandardScaler + OneHotEncoder → RandomizedSearchCV
         ↓                           ↓                              ↓
    Stratified Sampling          Pipeline Transform          Hyperparameter Tuning
         ↓                           ↓                              ↓
    Classification Models      Regression Models              Best Model Selection
    (Delay Yes/No)            (Delay Duration)                (LightGBM Regressor)
```

### Evaluation Framework
```
Standard Metrics (MAE, RMSE, R²) → Custom OAI Metric → SHAP Explainability
           ↓                              ↓                    ↓
    Model Comparison              Controllable vs External    Feature Importance
           ↓                              ↓                    ↓
    Best Model: LightGBM          68% Controllable           Top 5 Features Identified
```

## 💡 Operational Adjustability Index (OAI)

Custom metric prioritizing controllable delays for business impact:

```python
OAI_WEIGHTS = {
    'carrier_delay': 0.4,        # 40% - Airline operations
    'late_aircraft_delay': 0.4,  # 40% - Aircraft turnaround
    'weather_delay': 0.1,        # 10% - External factor
    'nas_delay': 0.1            # 10% - Air traffic control
}
```

**Why OAI?** Traditional metrics treat all delays equally. OAI focuses on **actionable delays** where airlines can intervene, making predictions more valuable for operational decision-making.

## 📈 Business Recommendations

Based on model insights and OAI analysis:

### High Priority Actions (15-20% Delay Reduction Potential)
1. **Aircraft Turnaround Optimization**
   - Add 15-20 minute buffers for connecting flights
   - Implement predictive maintenance protocols
   - Create rapid response teams

2. **Carrier-Specific Interventions**
   - Audit underperforming carriers (identified via model)
   - Benchmark best practices from top performers
   - Establish performance KPIs

3. **Seasonal Resource Planning**
   - Pre-position backup aircraft during peak months (June-July)
   - Increase staffing levels by 20% during high-delay periods
   - Enhanced weather contingency protocols



## 📦 Dependencies

- **pandas**, **numpy** - Data manipulation
- **scikit-learn** - Machine learning pipeline
- **LightGBM** - Gradient boosting (R²=0.82)
- **SHAP** - Model explainability
- **matplotlib**, **seaborn** - Visualization

## 🔬 Methodology

1. **Data Preprocessing** (110K+ records, 30+ airports)
   - Per-flight normalization to handle varying flight volumes
   - IQR-based outlier clipping for robust statistics
   - Frequency encoding for high-cardinality features (airport, carrier)

2. **Feature Engineering**
   - Delay components: carrier, late aircraft, weather, NAS
   - Temporal features: month, seasonality patterns
   - Aggregate metrics: delay counts, cancellation rates

3. **Model Selection via RandomizedSearchCV**
   - Tested: Linear Regression, Random Forest, XGBoost, **LightGBM** ✓
   - 5-fold cross-validation
   - Optimized for MAE with OAI constraints

4. **Explainability with SHAP**
   - TreeExplainer for LightGBM interpretability
   - Feature importance ranking
   - Individual prediction explanations

5. **Business Translation**
   - OAI metric design (68% controllable identification)
   - Actionable recommendation generation
   - Cost-benefit analysis (15-20% reduction potential)

## 📊 Results Highlights

### Model Comparison
| Model | R² | MAE (min) | RMSE (min) | OAI |
|-------|-----|-----------|------------|-----|
| Linear Regression | 0.73 | 2.45 | 4.12 | 9.2 |
| Random Forest | 0.79 | 1.85 | 3.21 | 8.1 |
| XGBoost | 0.81 | 1.62 | 2.95 | 7.8 |
| **LightGBM** ⭐ | **0.82** | **1.20** | **2.10** | **7.4** |

### SHAP Feature Importance
1. `late_aircraft_ct` - 28% impact
2. `carrier_ct` - 22% impact
3. `nas_delay` - 15% impact
4. `weather_ct` - 12% impact
5. `arr_del15` - 10% impact

## 🎓 Key Learnings

- **Cascading delays** (late aircraft) have the highest predictive power
- **Carrier operations** are second-most important and fully controllable
- **68% of delay factors** can be mitigated through operational improvements
- **Seasonal patterns** (June-July) require proactive resource allocation
- **Custom metrics** (OAI) align ML models with business objectives


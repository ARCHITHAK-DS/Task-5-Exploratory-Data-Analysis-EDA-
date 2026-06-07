# 🚢 Titanic Dataset — Exploratory Data Analysis (EDA)

## 📌 Project Overview

This project performs a comprehensive **Exploratory Data Analysis (EDA)** on the Titanic passenger dataset from the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic). The goal is to extract meaningful insights about survival patterns through statistical summaries and rich visualizations — laying the groundwork for predictive modelling.


## 📊 Dataset Description

| Property | Value |
|----------|-------|
| Source | [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data) |
| File | `train.csv` |
| Shape | 891 rows × 12 columns |
| Target | `Survived` (0 = No, 1 = Yes) |

### Data Dictionary

| Variable | Definition | Key |
|----------|------------|-----|
| `Survived` | Survival outcome | 0 = No, 1 = Yes |
| `Pclass` | Ticket class (socioeconomic proxy) | 1 = 1st, 2 = 2nd, 3 = 3rd |
| `Sex` | Passenger gender | male / female |
| `Age` | Age in years (fractional if < 1) | 177 missing values |
| `SibSp` | # of siblings / spouses aboard | — |
| `Parch` | # of parents / children aboard | — |
| `Ticket` | Ticket number | — |
| `Fare` | Passenger fare (£) | — |
| `Cabin` | Cabin number | 737 missing values |
| `Embarked` | Port of embarkation | C=Cherbourg, Q=Queenstown, S=Southampton |

---

## 🔧 Setup & Requirements

### Prerequisites

```bash
Python >= 3.8
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn reportlab jupyter
```

### Run the Notebook

```bash
# Clone / download the project, then:
cd titanic_eda
jupyter notebook Titanic_EDA.ipynb
```

## 🔑 Key Findings

### 1. Overall Survival
- Only **38.4%** of 891 passengers survived
- Dataset is moderately imbalanced → use F1/AUC-ROC, not accuracy

### 2. Gender is the Strongest Predictor
| Group | Survival Rate |
|-------|--------------|
| Female | **71.3%** |
| Male | **21.1%** |

"Women and children first" protocol clearly reflected in data.

### 3. Passenger Class Gradient
| Class | Survival Rate |
|-------|--------------|
| 1st Class | **63.0%** |
| 2nd Class | **42.4%** |
| 3rd Class | **26.9%** |

Higher class → better deck access → more lifeboat access.

### 4. Age Patterns
- Children **under 16** had ~52% survival
- Median age: **30.5 years** (19.9% of ages missing)
- 1st-class passengers skew older; 3rd-class skews younger

### 5. Fare as a Wealth Proxy
- Fare is heavily right-skewed (median £14.2, max £610)
- Positive correlation with survival: **r ≈ +0.26**
- Log-transformation recommended before modelling

### 6. Feature Correlations (with Survived)
| Feature | r |
|---------|---|
| Pclass | **-0.34** (strongest negative) |
| Fare | **+0.26** (strongest positive) |
| Age | -0.07 |
| SibSp | -0.04 |

### 7. Family Size
- **Small families (2–4)** had the best survival rates
- **Solo travellers** and **large families (7+)** fared worst
- Engineer `FamilySizeGroup`: Alone / Small / Large

### 8. Embarkation Port
- **Cherbourg**: ~55% survival (higher 1st-class share)
- **Southampton**: ~34% survival (most 3rd-class passengers)
- Effect is largely a **proxy for passenger class**

---

## 🔮 Recommendations for Modelling

### Feature Engineering

```python
# 1. Encode gender
df['Sex_encoded'] = df['Sex'].map({'female': 1, 'male': 0})

# 2. Family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['FamilySizeGroup'] = pd.cut(df['FamilySize'], bins=[0,1,4,20],
                                labels=['Alone','Small','Large'])

# 3. Cabin flag
df['HasCabin'] = df['Cabin'].notna().astype(int)

# 4. Age imputation
df['Age'] = df.groupby(['Sex','Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median()))

# 5. Log-transform Fare
import numpy as np
df['Fare_log'] = np.log1p(df['Fare'])

# 6. Fill Embarked
df['Embarked'] = df['Embarked'].fillna('S')

# 7. Extract title from Name
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
```

### Feature Importance Ranking (from EDA)
```
Sex > Pclass > Fare > Age > FamilySize > Embarked > SibSp/Parch
```

### Model Selection
```python
from sklearn.linear_model import LogisticRegression       # baseline
from sklearn.ensemble import RandomForestClassifier        # strong performer
from sklearn.ensemble import GradientBoostingClassifier    # best accuracy
from xgboost import XGBClassifier                          # production choice
```

### Evaluation Strategy
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 🛠 Tools & Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 1.5 | Data loading, `.describe()`, `.info()`, `.value_counts()` |
| `numpy` | ≥ 1.23 | Numerical operations |
| `matplotlib` | ≥ 3.6 | Histograms, bar charts, scatterplots |
| `seaborn` | ≥ 0.12 | `sns.heatmap()`, `sns.countplot()` |
| `reportlab` | ≥ 3.6 | PDF report generation |

---

## 📋 Visualizations Summary

| Figure | Type | Key Insight |
|--------|------|-------------|
| Fig 1 | Horizontal bar | Cabin 83% missing, Age 20% missing |
| Fig 2 | Pie + bar | 38.4% survived |
| Fig 3 | Bar charts | Women 71% vs Men 21%; 1st 63% vs 3rd 27% |
| Fig 4 | Overlapping histograms | Children had higher survival; 1st class older |
| Fig 5 | Histograms | Fare heavily right-skewed; class differences large |
| Fig 6 | Heatmap | Pclass strongest negative; Fare strongest positive |
| Fig 7 | Bar charts | Small families (2–4) survived best |
| Fig 8 | Bar + grouped bar | Cherbourg best; effect confounded by class |
| Fig 9 | 2×2 scatterplots | High fare + older → survived cluster |
| Fig 10 | Bar charts | SibSp=1 and Parch=1–3 optimal; extremes bad |

---
## 📌 Conclusion

The analysis reveals that gender, passenger class, and fare were the most important factors affecting survival on the Titanic.

The project demonstrates how Exploratory Data Analysis helps uncover patterns, relationships, and meaningful insights from raw data.


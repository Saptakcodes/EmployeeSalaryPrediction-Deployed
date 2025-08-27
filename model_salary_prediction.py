# ---------------------------------------------
# Employee Salary Prediction – Improved Version
# ---------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# ------------------- Load & Inspect -------------------
df = pd.read_csv("employee_salary_prediction.csv")
print(df.info())
print(df.head())

# ------------------- Basic Cleaning -------------------
df.fillna(df.median(numeric_only=True), inplace=True)
df.dropna(inplace=True)

# ------------------- Outlier Removal ------------------
for col in ['Salary', 'Years of Experience']:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# ------------------- Rare Job Titles ------------------
rare_jobs = df['Job Title'].value_counts()[lambda s: s < 5].index
df['Job Title'] = df['Job Title'].replace(rare_jobs, 'Other')

# -------------- Feature Engineering ------------------
# 1️⃣ Seniority (text → label)
def seniority(text: str) -> str:
    t = text.lower()
    if 'intern' in t:
        return 'Intern'
    if 'junior' in t:
        return 'Junior'
    if 'senior' in t:
        return 'Senior'
    if 'lead' in t or 'head' in t or 'principal' in t:
        return 'Lead'
    return 'Mid'

df['Seniority'] = df['Job Title'].apply(seniority)
seniority_map = {'Intern': 0, 'Junior': 1, 'Mid': 2, 'Senior': 3, 'Lead': 4}
df['SeniorityNum'] = df['Seniority'].map(seniority_map)

# 2️⃣ Job Category (manual mapping)
engineering = ['software', 'developer', 'engineer', 'network', 'devops', 'data']
sales       = ['sales', 'account', 'business development']
hr          = ['hr', 'human resources', 'recruit', 'talent']
marketing   = ['marketing', 'brand', 'social', 'seo', 'content', 'copy']
finance     = ['finance', 'accountant', 'analyst', 'controller']
def job_cat(title: str) -> str:
    t = title.lower()
    if any(k in t for k in engineering): return 'Engineering'
    if any(k in t for k in sales):       return 'Sales'
    if any(k in t for k in hr):          return 'HR'
    if any(k in t for k in marketing):   return 'Marketing'
    if any(k in t for k in finance):     return 'Finance'
    return 'Other'

df['Job Category'] = df['Job Title'].apply(job_cat)

# 3️⃣ Experience bucket
df['ExpBucket'] = pd.cut(
    df['Years of Experience'],
    bins=[-1, 2, 5, 10, 20, np.inf],
    labels=['0‑2', '3‑5', '6‑10', '11‑20', '21+']
)

# 4️⃣ Interaction feature Age × Experience
df['AgeExp'] = df['Age'] * df['Years of Experience']

# ---------------- Target & Features -------------------
y = np.log1p(df['Salary'])
X = df.drop(columns=['Salary'])

categorical_features = [
    'Gender', 'Education Level', 'Job Title', 'Seniority',
    'Job Category', 'ExpBucket'
]
numerical_features = [
    'Age', 'Years of Experience', 'SeniorityNum', 'AgeExp'
]

# -------------- Preprocessor & Pipeline --------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# ------------------- Train/Test -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = np.expm1(pipeline.predict(X_test))
y_test_orig = np.expm1(y_test)
print("\n📊 Initial R²:", f"{r2_score(y_test_orig, y_pred):.4f}")

# -------------- Hyper‑Parameter Search ---------------
param_dist = {
    'regressor__n_estimators': [200, 400, 600],
    'regressor__max_depth': [3, 5, 8],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__subsample': [0.7, 0.9, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0],
    'regressor__min_child_weight': [1, 3, 5],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
search.fit(X_train, y_train)

# ------------------- Evaluate -------------------------
best_pipe = search.best_estimator_
final_pred = np.expm1(best_pipe.predict(X_test))

mae  = mean_absolute_error(y_test_orig, final_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, final_pred))
r2   = r2_score(y_test_orig, final_pred)

print("\n📊 FINAL METRICS")
print(f"MAE : {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²  : {r2:.4f}")
print("\nBest CV R² :", search.best_score_)
print("Best Params:", search.best_params_)

# ------------------- Save Model -----------------------
joblib.dump(best_pipe, "salary_prediction_model_xgb.pkl")
print("\n✅  Model saved to salary_prediction_model_xgb.pkl")

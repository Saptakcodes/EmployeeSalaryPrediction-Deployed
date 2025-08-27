from flask import Flask, request, jsonify, render_template
import numpy as np, pandas as pd, joblib

app = Flask(__name__)
model = joblib.load("salary_prediction_model_xgb.pkl")

# ----- helpers (exact same logic used in training) -----
def seniority(txt):
    t = txt.lower()
    if 'intern' in t: return 'Intern'
    if 'junior' in t: return 'Junior'
    if 'senior' in t: return 'Senior'
    if any(k in t for k in ['lead','head','principal']): return 'Lead'
    return 'Mid'

def job_cat(t):
    t = t.lower()
    if any(k in t for k in ['software','developer','engineer','network','devops','data']): return 'Engineering'
    if any(k in t for k in ['sales','account','business development']): return 'Sales'
    if any(k in t for k in ['hr','human resources','recruit','talent']): return 'HR'
    if any(k in t for k in ['marketing','brand','social','seo','content','copy']): return 'Marketing'
    if any(k in t for k in ['finance','accountant','analyst','controller']): return 'Finance'
    return 'Other'

def exp_bucket(x):
    x = float(x)
    if x <= 2:  return '0‑2'
    if x <= 5:  return '3‑5'
    if x <= 10: return '6‑10'
    if x <= 20: return '11‑20'
    return '21+'
# -------------------------------------------------------

JOB_TITLES = pd.read_csv('employee_salary_prediction.csv')["Job Title"].fillna("Other").str.title().unique().tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/job_titles')
def job_titles():
    return jsonify(sorted(JOB_TITLES))

@app.route('/predict', methods=['POST'])
def predict():
    d = request.get_json()

    df = pd.DataFrame([{
        'Age':                 float(d['age']),
        'Gender':              d['gender'].capitalize(),
        'Education Level':     d['education'],          # adjust mapping if needed
        'Job Title':           d['jobTitle'].title(),
        'Years of Experience': float(d['experience'])
    }])

    # reproduce engineered features
    df['Seniority']      = df['Job Title'].apply(seniority)
    df['SeniorityNum']   = df['Seniority'].map({'Intern':0,'Junior':1,'Mid':2,'Senior':3,'Lead':4})
    df['Job Category']   = df['Job Title'].apply(job_cat)
    df['ExpBucket']      = df['Years of Experience'].apply(exp_bucket)
    df['AgeExp']         = df['Age'] * df['Years of Experience']

    pred_log = model.predict(df)[0]
    salary = float(round(np.expm1(pred_log), 2))


    return jsonify({'salary': salary, 'confidence': 0.78})


if __name__ == '__main__':
    app.run(debug=True)


# 💼 Salary Prediction Web App

This project is a **Machine Learning-powered web application** that predicts the **salary range** based on a user’s inputs such as education, experience, company size, and job title. The app is built using **Python, Scikit-learn, and Streamlit**, and deployed for public access.

---

## 🚀 Problem Statement

The tech industry offers a wide variety of roles with varying compensation. However, there is often a lack of clarity for job seekers and freshers regarding the **expected salary range** for a specific job role based on their qualifications, skills, and other parameters. This project aims to build a **salary prediction system** using **machine learning** to assist users in estimating salaries across different job profiles, aiding in **career planning and negotiation**.

---

## 🧠 System Approach

This system follows an **end-to-end machine learning pipeline**, which includes:

- **Data Collection** – Dataset sourced from Kaggle, containing job title, experience, education level, company size, etc.
- **Data Preprocessing** – Handling missing values, encoding categorical variables, and feature selection.
- **Model Training** – Using the **Random Forest Regressor** for robust performance on numerical predictions.
- **Model Evaluation** – Measuring R² Score, MAE, and RMSE.
- **Web Deployment** – Deploying the model using **Streamlit**, allowing users to interact with the model via a simple UI.

---

## 🧰 Tech Stack and Requirements

### ✅ Programming Language
- **Python 3.x**

### ✅ Libraries Used
- `pandas` – Data manipulation
- `numpy` – Numerical computations
- `matplotlib` / `seaborn` – Data visualization
- `scikit-learn` – Machine learning modeling
- `streamlit` – Web interface for deployment
- `pickle` – Model serialization

### ✅ Development Tools
- Jupyter Notebook / VS Code / Google Colab
- Git & GitHub for version control

---



### 💻 Tech Stack

- **Frontend**: HTML, CSS, TailwindCSS, JavaScript
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib (for offline evaluation)
- **Deployment**: Flask server (can be extended with Gunicorn + Nginx or hosted via platforms like Render or Heroku)

---

## 🔍 Algorithm Used

### 🔁 Random Forest Regressor
- A powerful ensemble learning method that builds multiple decision trees and averages the results for improved accuracy.
- Benefits: Handles nonlinear relationships and works well with both numerical and categorical data.
- Random Forest Regressor -> Supervised Learning Used

---

## 🌐 Deployment

- The trained model was serialized using **Pickle**.
- The web app was developed and deployed using **Streamlit**, offering:
  - Interactive UI for user inputs
  - Real-time prediction display
  - Mobile and desktop-friendly layout

---

<img width="1913" height="955" alt="image" src="https://github.com/user-attachments/assets/0a68cc66-7851-434d-b6fe-ac90896450e2" />
<img width="1918" height="835" alt="image" src="https://github.com/user-attachments/assets/10bbe0b3-44f0-4c94-950b-7a3f8bc0c94c" />
<img width="1838" height="855" alt="image" src="https://github.com/user-attachments/assets/51bf1217-a04c-4146-92e2-31758127dece" />
<img width="1898" height="828" alt="image" src="https://github.com/user-attachments/assets/a7f168f5-fe5d-4706-a1aa-bf3dc3ff4dff" />


---

## ✅ How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/salary-prediction-app.git
   cd salary-prediction-app

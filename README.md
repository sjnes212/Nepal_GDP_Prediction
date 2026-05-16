# Nepal GDP Prediction

A machine learning web application that predicts Nepal's GDP 
based on economic indicators using multiple regression models.

## Features
- Compares 3 ML models: Linear Regression, Random Forest, 
  and Support Vector Regression (SVR)
- Interactive web interface to input economic parameters
- Displays Model Evaluation Results (MAE and R² Score)
- Visualizes GDP trends over time

## Technologies Used
- Python
- Scikit-learn (Machine Learning)
- Flask (Web Framework)
- HTML/CSS (Frontend)
- Pandas & NumPy (Data Processing)

## Input Parameters
- Year
- Population (in millions)
- Inflation Rate (%)
- Unemployment Rate (%)

## Model Performance
| Model | R² Score |
|---|---|
| Linear Regression | 0.842 |
| Random Forest | 0.999 |
| Support Vector Regression | ~0.00 |

## How to Run
```bash
pip install -r requirements.txt
python app.py
```
Then open: http://127.0.0.1:5000/predict

## Dataset
Historical Nepal economic data (nepal_gdp.csv)

## Developer
Jenish Silwal — Bachelor of Computer Information Systems,  
Pokhara University, Nepal

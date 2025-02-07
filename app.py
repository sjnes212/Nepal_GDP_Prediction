import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv('nepal_gdp.csv')

# Prepare the data
X = df[['Year', 'Population (in millions)', 'Inflation Rate (%)', 'Unemployment Rate (%)']]
y = df['GDP (in billion USD)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Support Vector Regression': SVR()
}

# Create a static folder if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Plot the actual GDP data
plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['GDP (in billion USD)'], color='blue', label='Actual GDP')
plt.xlabel('Year')
plt.ylabel('GDP (in billion USD)')
plt.title('Nepal GDP Over Time')
plt.legend()
plt.savefig('static/gdp_plot.png')  # Save the plot as an image
plt.close()

# Evaluate models and store results
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'R2 Score': r2}

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', results=results)

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    population = float(request.form['population'])
    inflation = float(request.form['inflation'])
    unemployment = float(request.form['unemployment'])
    
    input_data = np.array([[year, population, inflation, unemployment]])
    
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(input_data)[0]
    
    return render_template('index.html', results=results, predictions=predictions, plot_url='static/gdp_plot.png')

if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Load hourly data
df_hour = pd.read_csv(r'C:\Users\Nivetha\OneDrive\Desktop\bike_sharing\data\bike+sharing+dataset\hour.csv')
df_hour = pd.get_dummies(df_hour, columns=['season', 'weathersit'], drop_first=True)
df_numeric = df_hour.select_dtypes(include=[float, int])

# Feature Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df_numeric.corr()
sns.heatmap(corr, annot=True, cmap='Spectral', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# Feature selection and model training
features = ['temp', 'hum', 'hr', 'holiday', 'workingday', 'season_2', 'season_3', 'season_4', 'weathersit_2', 'weathersit_3']
X = df_hour[features]
y = df_hour['cnt']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Train Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Predictions
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

# Evaluate models
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f'Linear Regression RMSE: {rmse_lr}')
print(f'Random Forest RMSE: {rmse_rf}')

# Plot Actual vs Predicted Rentals for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='blue', edgecolor='k', label='Random Forest Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Perfect Prediction Line')
plt.title('Actual vs Predicted Bike Rentals (Random Forest)', fontsize=16)
plt.xlabel('Actual Rentals', fontsize=14)
plt.ylabel('Predicted Rentals', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Plot Actual vs Predicted Rentals for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.6, color='green', edgecolor='k', label='Linear Regression Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Perfect Prediction Line')
plt.title('Actual vs Predicted Bike Rentals (Linear Regression)', fontsize=16)
plt.xlabel('Actual Rentals', fontsize=14)
plt.ylabel('Predicted Rentals', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Bar plot for Average Rentals by Hour
df_hour.groupby('hr')['cnt'].mean().plot(kind='bar', color='skyblue', figsize=(10, 6))
plt.title('Average Bike Rentals by Hour of the Day', fontsize=16)
plt.xlabel('Hour of the Day', fontsize=14)
plt.ylabel('Average Rentals', fontsize=14)
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# Scatter plot for Rentals vs. Temperature
plt.figure(figsize=(10, 6))
plt.scatter(df_hour['temp'], df_hour['cnt'], alpha=0.5, color='purple')
plt.title('Bike Rentals vs. Temperature', fontsize=16)
plt.xlabel('Temperature (Normalized)', fontsize=14)
plt.ylabel('Number of Rentals', fontsize=14)
plt.grid(True)
plt.show()

# Actual vs Predicted Rentals Table
result_table = pd.DataFrame({
    'Actual Rentals': y_test[:10],
    'Predicted Rentals (Random Forest)': y_pred_rf[:10],
    'Predicted Rentals (Linear Regression)': y_pred_lr[:10]
})
print("\nActual vs. Predicted Rentals (Sample Table):")
print(result_table)

# Plot comparison of actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(range(10), y_test[:10], marker='o', color='blue', label='Actual Rentals')
plt.plot(range(10), y_pred_rf[:10], marker='x', color='red', linestyle='--', label='Predicted Rentals (Random Forest)')
plt.plot(range(10), y_pred_lr[:10], marker='x', color='green', linestyle='--', label='Predicted Rentals (Linear Regression)')
plt.title('Comparison of Actual vs Predicted Bike Rentals (First 10 Instances)', fontsize=16)
plt.xlabel('Instance', fontsize=14)
plt.ylabel('Number of Rentals', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Interactive Dashboard using Dash

# Initialize the Dash app
app = dash.Dash(__name__)

# Create the layout for the dashboard
app.layout = html.Div([
    html.H1("Bike Sharing Prediction Dashboard"),
    
    # Drop-down to select the model
    html.Label("Choose Model"),
    dcc.Dropdown(
        id='model_dropdown',
        options=[
            {'label': 'Linear Regression', 'value': 'lr'},
            {'label': 'Random Forest', 'value': 'rf'}
        ],
        value='rf'
    ),
    
    # Graph output
    dcc.Graph(id='predicted_vs_actual_graph'),

    # Data table
    html.H2("Actual vs Predicted Rentals (First 10 Samples)"),
    dcc.Graph(
        id='result_table',
        figure=px.line(result_table, title='Actual vs Predicted Rentals')
    )
])

# Define the callback for interactivity
@app.callback(
    Output('predicted_vs_actual_graph', 'figure'),
    Input('model_dropdown', 'value')
)
def update_graph(selected_model):
    if selected_model == 'lr':
        figure = px.scatter(x=y_test[:10], y=y_pred_lr[:10],
                            labels={'x': 'Actual Rentals', 'y': 'Predicted Rentals'},
                            title='Linear Regression Predictions vs Actual')
    else:
        figure = px.scatter(x=y_test[:10], y=y_pred_rf[:10],
                            labels={'x': 'Actual Rentals', 'y': 'Predicted Rentals'},
                            title='Random Forest Predictions vs Actual')
    return figure

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

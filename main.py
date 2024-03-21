""" Main script for implementing basic 10-day ahead consumption forecasts for Reel
    by David Ribberholt Ipsen.
    The script compares three different moodels:
        1. A mean model
        2. Linear regression model
        3. XGBoost model
"""
# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load data
df = pd.read_csv('dataset.csv')
df = df.iloc[::-1].reset_index(drop=True)
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
df['Datetime'] = df['Datetime'].dt.tz_convert('Europe/Copenhagen')

# Plot data
px.line(df, x='Datetime', y='Sum of Quantity', title='Consumption of Electricity')

# Feature Engineering
df['Month'] = df['Datetime'].dt.month
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Lag240'] = df['Sum of Quantity'].shift(240)

# One-hot encode categorical features
df = pd.concat([df, pd.get_dummies(df['Month'], prefix='Month', drop_first=True)], axis=1)
df = pd.concat([df, pd.get_dummies(df['Hour'], prefix='Hour', drop_first=True)], axis=1)
df = pd.concat([df, pd.get_dummies(df['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)
remove_cols = ['Month', 'Hour', 'DayOfWeek']
df = df.drop(remove_cols, axis=1)

# Split the data into training and test sets
X = df.drop(['Sum of Quantity', 'Datetime'], axis=1)[240:].values
y = df['Sum of Quantity'][240:].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


#################################################################################
################            Linear Regression                ####################
#################################################################################
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the the mapping between coefficients and features
print('Intercept: ', model.intercept_)
for i, col in enumerate(df.drop(['Sum of Quantity', 'Datetime'], axis=1).columns):
    print(col, ': ', model.coef_[i])

# Make predictions and evaluate
y_pred_lm = model.predict(X_test)
print('RMSE: ', np.sqrt(np.mean((y_test - y_pred_lm)**2)))
print('MAE: ', np.mean(np.abs(y_test - y_pred_lm)))
print('R^2: ', model.score(X_test, y_test))

# Plot the predictions and targets in the same plotly plot.
def plot_predictions(y_test, y_pred, period_of_train_to_plot, additional_series=None):
    """Plot the predictions and targets in the same plotly plot."""
    fig = go.Figure()

    fig.update_layout(
        title=dict(text="Forecasts of electricity consumption in test period")
    )

    # Add training data
    fig.add_trace(go.Scatter(x=df['Datetime'][-len(y_test)-period_of_train_to_plot:-len(y_test)],
                             y=df['Sum of Quantity'][-len(y_test)-period_of_train_to_plot:-len(y_test)],
                             name='Train',
                             line=dict(color="cornflowerblue",width=2), opacity=0.5))

    # Add test data
    fig.add_trace(go.Scatter(x=df['Datetime'][-len(y_test):],
                             y=y_test,
                             name='Test',
                             line=dict(color="cornflowerblue",width=2)))

    # Add predictions
    fig.add_trace(go.Scatter(x=df['Datetime'][-len(y_test):],
                             y=y_pred,
                             name='Predictions'))

    # Add additional series
    if additional_series:
        for series_name, series_data in additional_series.items():
            fig.add_trace(go.Scatter(x=df['Datetime'][-len(y_test):],
                                     y=series_data,
                                     name=series_name,
                                     line=dict()))
    fig.show()

period_of_train_to_plot = 30*24
plot_predictions(y_test, y_pred_lm, period_of_train_to_plot)


#################################################################################
################                XGBoost                      ####################
#################################################################################

# Fit XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Plot feature importances
if False:
    xgb.plot_importance(xgb_model)
    plt.show()

# Evaluate
print('RMSE: ', np.sqrt(np.mean((y_test - y_pred_xgb)**2)))
print('MAE: ', np.mean(np.abs(y_test - y_pred_xgb)))
print('R^2: ', xgb_model.score(X_test, y_test))

# Plot predictions
plot_predictions(y_test, y_pred_xgb, period_of_train_to_plot)


#################################################################################
################            BASELINE - MEAN MODEL            ####################
#################################################################################
y_pred_mean = np.repeat(np.mean(y_train), y_test.shape[0])

# Evaluate
print('RMSE: ', np.sqrt(np.mean((y_test - y_pred_mean)**2)))
print('MAE: ', np.mean(np.abs(y_test - y_pred_mean)))
print('R^2: ', 1 - np.sum((y_test-y_pred_mean)**2)/np.sum((y_test-y_pred_mean)**2))




#################################################################################
################            Comparison of models             ####################
#################################################################################

# Make table of performance metrics
print(f"RMSE: \n \
    Mean    model: {(np.sqrt(np.mean((y_test - y_pred_mean)**2))).round(1)} \n \
    Linear  model: {(np.sqrt(np.mean((y_test - y_pred_lm)**2))).round(1)} \n \
    XGBoost model: {(np.sqrt(np.mean((y_test - y_pred_xgb)**2))).round(1)} \
    ")

print(f"MAE: \n \
    Mean    model: {(np.mean(np.abs((y_test - y_pred_mean)))).round(1)} \n \
    Linear  model: {(np.mean(np.abs((y_test - y_pred_lm)))).round(1)} \n \
    XGBoost model: {(np.mean(np.abs((y_test - y_pred_xgb)))).round(1)} \
    ")

print(f"R^2: \n \
    Mean    model: {(1 - np.sum((y_test-y_pred_mean)**2)/np.sum((y_test-y_pred_mean)**2)).round(2)} \n \
    Linear  model: {(1 - np.sum((y_test-y_pred_lm)**2)/np.sum((y_test-y_pred_mean)**2)).round(2)} \n \
    XGBoost model: {(1 - np.sum((y_test-y_pred_xgb)**2)/np.sum((y_test-y_pred_mean)**2)).round(2)} \
    ")

# Plot predictions all together
plot_predictions(y_test, y_pred_mean, 14*24, {'Mean model': y_pred_mean,
                                              'Linear model': y_pred_lm,
                                              'XGBoost model': y_pred_xgb})





# TO IMPLEMENT:
# 1. At t=0+h, use latest measurement (not 10 days old)
# 2. Hyperparameter tuning
# 3. Walk-forward cross-validation

# TO ADD:
# 4. Add weather data
# 5. Add holidays
# 6. Add real-time measurements (if possible)
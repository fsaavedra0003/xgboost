# src/train_model.py
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("data/stock_data.csv", parse_dates=["Date"], index_col="Date")

# Feature engineering
df["Return"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_20"] = df["Close"].rolling(window=20).mean()
df["Volatility"] = df["Close"].rolling(window=10).std()
df.dropna(inplace=True)

# Target is next-day close price
df["Target"] = df["Close"].shift(-1)
df.dropna(inplace=True)

# Features and target
features = ["Open", "High", "Low", "Close", "Volume", "Return", "MA_5", "MA_20", "Volatility"]
X = df[features]
y = df["Target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train XGBoost
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test.values, label="Actual")
plt.plot(y_test.index, y_pred, label="Predicted")
plt.legend()
plt.title("Stock Price Prediction with XGBoost")
plt.xlabel("Date")
plt.ylabel("Price")
os.makedirs("results", exist_ok=True)
plt.savefig("results/prediction_plot.png")
plt.show()

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# === File list ===
excel_files = [
    r"D:\BS Computer Science Materials\USTP\3rd Year 2nd Semester\CS324 Machine Learning\Stock Prediction Model\Amazon_AMZN.csv.xlsx",
    r"D:\BS Computer Science Materials\USTP\3rd Year 2nd Semester\CS324 Machine Learning\Stock Prediction Model\Apple_AAPL.csv.xlsx",
    r"D:\BS Computer Science Materials\USTP\3rd Year 2nd Semester\CS324 Machine Learning\Stock Prediction Model\Google_GOOGL.csv.xlsx",
    r"D:\BS Computer Science Materials\USTP\3rd Year 2nd Semester\CS324 Machine Learning\Stock Prediction Model\Meta_META.csv.xlsx",
    r"D:\BS Computer Science Materials\USTP\3rd Year 2nd Semester\CS324 Machine Learning\Stock Prediction Model\Netflix_NFLX.csv.xlsx",
    r"D:\BS Computer Science Materials\USTP\3rd Year 2nd Semester\CS324 Machine Learning\Stock Prediction Model\Nvidia_NVDA.csv.xlsx",
    r"D:\BS Computer Science Materials\USTP\3rd Year 2nd Semester\CS324 Machine Learning\Stock Prediction Model\Tesla_TSLA.csv.xlsx",
]

# === Technical indicators ===
def add_technical_indicators(df):
    df['MA7'] = df['Close'].rolling(window=7).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['DailyReturn'] = df['Close'].pct_change()
    return df

# === Load and preprocess ===
combined_df = pd.DataFrame()
for file in excel_files:
    try:
        df = pd.read_excel(file)
        filename = os.path.basename(file)
        symbol = os.path.splitext(os.path.splitext(filename)[0])[0]

        required_cols = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Missing columns in {file}: {missing}")
            continue

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.sort_values('Date', inplace=True)
        df['Stock'] = symbol

        df = add_technical_indicators(df)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        combined_df = pd.concat([combined_df, df], ignore_index=True)
        print(f"Loaded and processed: {symbol} — {len(df)} rows")
    except Exception as e:
        print(f"Failed to process {file}: {e}")

# === Feature engineering ===
combined_df['Year'] = combined_df['Date'].dt.year
combined_df['Month'] = combined_df['Date'].dt.month
combined_df['Day'] = combined_df['Date'].dt.day
le = LabelEncoder()
combined_df['StockEncoded'] = le.fit_transform(combined_df['Stock'])

feature_cols = ['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day', 'StockEncoded', 'MA7', 'RSI', 'DailyReturn']
X = combined_df[feature_cols]
y = combined_df['Close']

# === Normalize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# === Traditional Models ===
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

print("\nCombined Dataset Model Evaluation:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(y_test.values, label='Actual Close Price')
    plt.plot(y_pred, label='Predicted Close Price')
    plt.title(f"{name} Predictions vs Actual")
    plt.xlabel("Test Data Points")
    plt.ylabel("Stock Close Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    avg_price = y_test.mean()
    error_pct = (rmse / avg_price) * 100
    print(f"--> {name} RMSE is {rmse:.2f}, which is {error_pct:.2f}% of the average closing price.\n")

# === SVR Optimized Grid ===
print("Tuning SVR hyperparameters (optimized)...")

param_grid = {
    'kernel': ['rbf'],           # Best-performing kernel in most cases
    'C': [10, 100],              # Regularization strength
    'gamma': ['scale', 0.01],   # Kernel coefficient
    'epsilon': [0.01, 0.1]       # Tolerance margin
}

svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_svr = grid_search.best_estimator_
y_pred_svr = best_svr.predict(X_test)

rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

avg_price = y_test.mean()
error_pct_svr = (rmse_svr / avg_price) * 100

print("\nSupport Vector Regression (Optimized) Results:")
print(f"  RMSE: {rmse_svr:.2f}")
print(f"  MAE: {mae_svr:.2f}")
print(f"  R²: {r2_svr:.4f}")
print(f"--> SVR RMSE is {rmse_svr:.2f}, which is {error_pct_svr:.2f}% of the average closing price.\n")

plt.figure(figsize=(10, 4))
plt.plot(y_test.values, label='Actual Close Price')
plt.plot(y_pred_svr, label='Predicted Close Price (Optimized SVR)')
plt.title("SVR Predictions vs Actual")
plt.xlabel("Test Data Points")
plt.ylabel("Stock Close Price")
plt.legend()
plt.tight_layout()
plt.show()

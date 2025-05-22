#  Stock Price Prediction System

This project is a machine learning-based stock price prediction system built in Python using `scikit-learn`. It uses historical data from Yahoo Finance for major tech stocks (Apple, Amazon, Google, Meta, Netflix, Nvidia, Tesla) and implements regression models to forecast future stock closing prices.

##  Project Objective

- To design and implement a regression-based prediction system for stock prices.
- To evaluate and compare model performance using metrics such as RMSE, MAE, and R².
- To gain insights from historical trends and visualize predictions against actual data.

##  Project Structure

All project files are located in the folder `Stock Prediction Model`. This includes:

- Raw and cleaned Excel datasets
- Python scripts for data processing, modeling, and evaluation
- Jupyter notebooks (if applicable)
- Graphs and visualizations of model predictions

##  Data Collection

Historical stock data was fetched using the `yfinance` Python package. The datasets include:
- Date
- Open
- High
- Low
- Close
- Volume

### Companies Analyzed:
- Apple (AAPL)
- Amazon (AMZN)
- Google (GOOGL)
- Meta (META)
- Netflix (NFLX)
- Nvidia (NVDA)
- Tesla (TSLA)

##  Data Preprocessing

Preprocessing steps included:
- Manual correction of Excel file headers
- Formatting date columns
- Dropping invalid rows
- Feature engineering (e.g., MA7, RSI, Daily Return)
- Encoding categorical stock labels using `LabelEncoder`
- Feature scaling with `StandardScaler`

##  Models Used

Three machine learning regression models were implemented:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Support Vector Regressor (SVR)**

> **Note:** SVR took significantly longer to train due to limited hardware (MX130 GPU, i5 10th-gen CPU).

##  Evaluation Results

Models were trained and tested on a combined dataset. Below are the final metrics:

| Model               | RMSE | MAE  | R²     | % of Avg Close Price |
|--------------------|------|------|--------|-----------------------|
| Linear Regression  | 1.29 | 0.50 | 0.9998 | 2.85%                 |
| Random Forest      | 1.33 | 0.42 | 0.9998 | 2.95%                 |
| SVR (Tuned)        | 1.48 | 0.66 | 0.9997 | 3.28%                 |

## Visual Results

The project includes graphs comparing actual vs. predicted closing prices for visual inspection and insight.

## ⚙️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/stock-price-predictor.git
   cd "Stock Prediction Model"

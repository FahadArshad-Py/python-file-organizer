import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pytz
import matplotlib.dates as mdates

Model_Path = 'models/gold_model.pkl'

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(data, fast=12, slow=26):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line

symbol = "GC=F"

print("Downloading data...")
data = yf.download(symbol, period="365d", interval="1h")

if data.empty:
    print("Failed to fetch data. Please check the symbol or internet connection.")
    exit()

close_prices = data['Close']

print("Calculating indicators...")
data['rsi'] = calculate_rsi(close_prices)
data['macd'] = calculate_macd(close_prices)
data['sma_5'] = close_prices.rolling(window=5).mean()
data['sma_10'] = close_prices.rolling(window=10).mean()
data['return_1d'] = close_prices.pct_change()
data['sma_ratio'] = data['sma_5'] / data['sma_10']
data['lag_1'] = close_prices.shift(1)
data['lag_2'] = close_prices.shift(2)

# Define target: significant price movement
threshold = 0.005  # 0.5% threshold
price_diff = close_prices.shift(-1) - close_prices
price_pct_change = price_diff / close_prices
data['target'] = np.where(price_pct_change > threshold, 1, 0)

# Clean data
data.dropna(inplace=True)

features = data[['rsi', 'macd', 'sma_5', 'sma_10', 'return_1d', 'sma_ratio', 'lag_1', 'lag_2']]
target = data['target']

# TimeSeriesSplit for time-aware validation
tscv = TimeSeriesSplit(n_splits=5)
train_index, test_index = list(tscv.split(features))[-1]
X_train, X_test = features.iloc[train_index], features.iloc[test_index]
y_train, y_test = target.iloc[train_index], target.iloc[test_index]

# Resample using SMOTE
print("Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

print("Training model...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
)
model.fit(X_train_resampled, y_train_resampled)

# Save the trained model
joblib.dump(model, Model_Path)
print("✅ Model trained and saved.")

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2%}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Define the GMT+5 timezone (Pakistan Standard Time, for example)
timezone = pytz.timezone('Asia/Karachi')

# Convert the index to the desired timezone (GMT+5)
if data.index.tz is None:
    print("Index is naive, localizing to UTC first.")
    data.index = data.index.tz_localize('UTC')  # Localize to UTC if it's naive
else:
    print("Index is already timezone-aware.")

# Now convert the timezone to Asia/Karachi (GMT+5)
data.index = data.index.tz_convert(timezone)  # Convert to GMT+5 (Asia/Karachi)

# Plotting actual vs predicted prices
plt.figure(figsize=(15, 6))
plt.plot(data.index, data['Close'], label='Gold Price', color='blue')

# Buy and sell signals
buy_signals = data.iloc[test_index].index[y_pred == 1]
sell_signals = data.iloc[test_index].index[y_pred == 0]

plt.scatter(buy_signals, data.loc[buy_signals, 'Close'], color='green', label='Buy Signal', marker='^', alpha=1)
plt.scatter(sell_signals, data.loc[sell_signals, 'Close'], color='red', label='Sell Signal', marker='v', alpha=1)

# Labels and formatting
plt.legend()
plt.title("Gold Price Buy/Sell Signals")
plt.xlabel("Date & Time")
plt.ylabel("Price")
plt.grid(True)

# Formatting the x-axis to show Date & Time in 12-hour format with AM/PM and GMT+5
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %I:%M %p'))  # 12-hour format with AM/PM
plt.tight_layout()

plt.show()

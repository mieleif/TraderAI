import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file with Ichimoku indicator values
source_file = 'ETHUSDT_4_hours_data.csv'
df = pd.read_csv(source_file, index_col=0, parse_dates=True)

# Create the new feature
df['prev_high'] = df['high'].shift(1)
df['prev_low'] = df['low'].shift(1)
df['laging_span'] = df['chikou_span'].shift(26)
df['laging_span_price'] = df['close'].shift(26)

# Create the daily target variable: 1 if next day's close is higher, 0 otherwise.
df['future_return'] = df['close'].shift(-1) - df['close']
df['target'] = (df['future_return'] > 0).astype(int)

# Create the weekly target variable: 1 if the close 7 days later is higher, 0 otherwise.
df['future_week_return'] = df['close'].shift(-7) - df['close']
df['target_week'] = (df['future_week_return'] > 0).astype(int)

# Drop rows with NaN values (due to shifting operations)
df.dropna(inplace=True)

# Example: preparing features and targets for the daily prediction model
features = ['open','prev_high', 'prev_low', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'laging_span', 'laging_span_price']
X = df[features]
y_daily = df['target']
y_weekly = df['target_week']

# Splitting the data into training and testing sets (without shuffling to maintain time order)
X_train, X_test, y_daily_train, y_daily_test = train_test_split(X, y_daily, test_size=0.2, shuffle=False)
_, _, y_weekly_train, y_weekly_test = train_test_split(X, y_weekly, test_size=0.2, shuffle=False)

# Example: training a model for the daily target (similar approach applies for weekly target)
daily_model = LogisticRegression()
daily_model.fit(X_train, y_daily_train)
daily_pred = daily_model.predict(X_test)
print("Daily Target Model Accuracy:", accuracy_score(y_daily_test, daily_pred))
print("Daily Classification Report:\n", classification_report(y_daily_test, daily_pred))

# Similarly, you can train a model for the weekly target:
weekly_model = LogisticRegression()
weekly_model.fit(X_train, y_weekly_train)
weekly_pred = weekly_model.predict(X_test)
print("Weekly Target Model Accuracy:", accuracy_score(y_weekly_test, weekly_pred))
print("Weekly Classification Report:\n", classification_report(y_weekly_test, weekly_pred))

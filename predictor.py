import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import MACD

def make_prediction(df):
    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['macd'] = MACD(df['close']).macd_diff()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Drop rows with any NaN values
    df = df.dropna()

    # Features and target must be same length
    X = df[['rsi', 'macd']]
    y = df['target']

    if len(X) < 10:
        return "Not enough data", 0

    model = RandomForestClassifier()
    model.fit(X, y)

    latest = X.iloc[-1:].values.reshape(1, -1)
    pred = model.predict(latest)[0]
    conf = round(model.predict_proba(latest)[0][pred] * 100, 2)
    return ("UP" if pred == 1 else "DOWN"), conf

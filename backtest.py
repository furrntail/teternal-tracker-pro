import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import MACD

def run_backtest(df):
    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['macd'] = MACD(df['close']).macd_diff()
    df = df.dropna()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()

    X = df[['rsi', 'macd']]
    y = df['target']
    model = RandomForestClassifier()
    model.fit(X, y)
    df['prediction'] = model.predict(X)
    accuracy = (df['prediction'] == df['target']).mean()
    return {"Accuracy": round(accuracy * 100, 2), "Total Records": len(df)}
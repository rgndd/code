import os
import pandas as pd
import numpy as np

def load_data():
    air = pd.read_csv('data/raw/air_quality.csv')
    return air

def preprocess_data(air):
    df = air.copy()
    df = df.fillna(method='ffill').fillna(method='bfill')
    df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5
    return df

def save_processed_data(df):
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/processed_data.csv', index=False)

def main():
    air = load_data()
    df = preprocess_data(air)
    save_processed_data(df)
    print('数据处理完成，已保存到 data/processed/processed_data.csv')

if __name__ == '__main__':
    main() 
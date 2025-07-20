import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import joblib
import requests
import datetime
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_pred_vs_true():
    df = pd.read_csv('results/pred_vs_true.csv')
    plt.figure(figsize=(6,6))
    plt.scatter(df['真实AQI'], df['预测AQI'], alpha=0.6)
    min_val = min(df['真实AQI'].min(), df['预测AQI'].min())
    max_val = max(df['真实AQI'].max(), df['预测AQI'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('真实AQI')
    plt.ylabel('预测AQI')
    plt.title('预测值与真实值对比')
    plt.tight_layout()
    plt.savefig('results/pred_vs_true.png')
    print('预测值与真实值对比图已保存到 results/pred_vs_true.png')

def plot_pred_distribution():
    df = pd.read_csv('results/pred_vs_true.csv')
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x='预测AQI', bins=30, kde=True, stat='probability')
    plt.xlabel('预测AQI')
    plt.ylabel('频率')
    plt.title('预测AQI分布')
    plt.tight_layout()
    plt.savefig('results/pred_distribution.png')
    print('预测AQI分布图已保存到 results/pred_distribution.png')

def plot_feature_importance():
    # 读取模型和特征
    df = pd.read_csv('data/processed/merged_data.csv')
    features = [
        'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'windspeed_10m_max',
        'month', 'season', 'week_of_year', 'day_of_year', 'dayofweek', 'is_weekend'
    ]
    features = [f for f in features if f in df.columns]
    model = joblib.load('results/best_model.pkl')
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.Series(importances, index=features).sort_values(ascending=True)
        plt.figure(figsize=(8,5))
        feature_importance.plot(kind='barh')
        plt.xlabel('重要性')
        plt.title('特征重要性')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        print('特征重要性图已保存到 results/feature_importance.png')
    else:
        print('当前模型不支持特征重要性输出')

def fetch_future_weather(city, lat, lon, days=7):
    today = datetime.date.today()
    start_date = today + datetime.timedelta(days=1)
    end_date = start_date + datetime.timedelta(days=days-1)
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
        f"&start_date={start_date}&end_date={end_date}&timezone=Asia/Shanghai"
    )
    resp = requests.get(url)
    data = resp.json()['daily']
    df = pd.DataFrame(data)
    df['city'] = city
    return df

def plot_future_aqi_forecast():
    # 城市及坐标
    city_coords = {
        '沈阳': (41.8, 123.4),
        '大连': (38.9, 121.6),
        '鞍山': (41.1, 122.9),
    }
    model = joblib.load('results/best_model.pkl')
    df_hist = pd.read_csv('data/processed/merged_data.csv')
    # 只用训练时的特征
    future_features = [
        'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'windspeed_10m_max',
        'month', 'season', 'week_of_year', 'day_of_year', 'dayofweek', 'is_weekend'
    ]
    all_pred = []
    for city, (lat, lon) in city_coords.items():
        df = fetch_future_weather(city, lat, lon, days=7)
        # 构造衍生特征
        df['dayofweek'] = pd.to_datetime(df['time']).dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
        df['month'] = pd.to_datetime(df['time']).dt.month
        df['season'] = pd.to_datetime(df['time']).dt.month % 12 // 3 + 1
        df['week_of_year'] = pd.to_datetime(df['time']).dt.isocalendar().week
        df['day_of_year'] = pd.to_datetime(df['time']).dt.dayofyear
        # 只保留future_features
        X_pred = df.rename(columns={'city':'location', 'time':'date'})
        for f in future_features:
            if f not in X_pred.columns:
                X_pred[f] = 0
        X_pred = X_pred[future_features]
        X_pred = X_pred.fillna(0)
        y_pred = model.predict(X_pred)
        df['预测AQI'] = y_pred
        df['date'] = df['time']
        all_pred.append(df[['date','city','预测AQI']])
    result = pd.concat(all_pred)
    plt.figure(figsize=(10,6))
    for city in city_coords:
        sub = result[result['city']==city]
        plt.plot(sub['date'], sub['预测AQI'], marker='o', label=city)
    plt.xlabel('日期')
    plt.ylabel('预测AQI')
    plt.title('未来7天AQI预测')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/future_aqi_forecast.png')
    print('未来7天AQI预测图已保存到 results/future_aqi_forecast.png')

def plot_all_cities_future_aqi():
    df_pred = pd.read_csv('results/future_7days_aqi_pred.csv')
    plt.figure(figsize=(12, 6))
    for city in df_pred['location'].unique():
        city_df = df_pred[df_pred['location'] == city]
        plt.plot(city_df['date'], city_df['AQI_pred'], marker='o', label=city)
    plt.xlabel('日期')
    plt.ylabel('预测AQI')
    plt.title('所有城市未来7天AQI预测')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('results/future_7days_aqi_pred.png')
    plt.close()
    print('所有城市未来7天AQI预测图已保存到 results/future_7days_aqi_pred.png')

def main():
    # plot_time_series()  # 移除AQI时序图
    plot_pred_vs_true()
    plot_pred_distribution()
    plot_feature_importance()
    plot_future_aqi_forecast()
    plot_all_cities_future_aqi()

if __name__ == '__main__':
    main() 
import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

city_coords = {
    '沈阳': (41.8, 123.4),
    '大连': (38.9, 121.6),
    '鞍山': (41.1, 122.9),
    '抚顺': (41.9, 123.9),
    '本溪': (41.3, 123.8),
    '丹东': (40.1, 124.4),
    '锦州': (41.1, 121.1),
    '营口': (40.7, 122.2),
    '阜新': (42.0, 121.7),
    '辽阳': (41.3, 123.2),
    '盘锦': (41.2, 122.0),
    '铁岭': (42.3, 123.8),
    '朝阳': (41.6, 120.4),
    '葫芦岛': (40.7, 120.8),
}

def get_weather(city, lat, lon, start_date, end_date):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
        f"&timezone=Asia/Shanghai"
    )
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['daily'])
    df['city'] = city
    return df

all_weather = []
for city, (lat, lon) in city_coords.items():
    print(f'Fetching weather for {city}...')
    df = get_weather(city, lat, lon, '2025-06-20', '2025-07-03')
    all_weather.append(df)
    time.sleep(1)  # 防止请求过快被限流

weather_df = pd.concat(all_weather, ignore_index=True)
weather_df.rename(columns={'city': 'location'}, inplace=True)
weather_df.to_csv('data/raw/weather_liaoning.csv', index=False)
print('已保存到 data/raw/weather_liaoning.csv')

aqi = pd.read_csv('data/processed/processed_data.csv')
weather = pd.read_csv('data/raw/weather_liaoning.csv')

# 统一日期字段名
weather = weather.rename(columns={'time': 'date'})
# 修正location字段，去掉"市"字
aqi['location'] = aqi['location'].str.replace('市', '', regex=False)
# 确保date字段为datetime类型
aqi['date'] = pd.to_datetime(aqi['date'])
weather['date'] = pd.to_datetime(weather['date'])
# 增加时间特征
if not pd.api.types.is_datetime64_any_dtype(aqi['date']):
    aqi['date'] = pd.to_datetime(aqi['date'])
# 月份、季节、周、年内天数
aqi['month'] = aqi['date'].dt.month
aqi['season'] = aqi['date'].dt.month % 12 // 3 + 1
aqi['week_of_year'] = aqi['date'].dt.isocalendar().week
aqi['day_of_year'] = aqi['date'].dt.dayofyear
# AQI滞后特征（前1-3天）
for i in range(1, 4):
    aqi[f'AQI_lag_{i}'] = aqi.groupby('location')['AQI'].shift(i)
# 打印合并前数据量
print('合并前AQI行数:', len(aqi))
print('合并前weather行数:', len(weather))
# 打印date和location字段前几行，检查格式
print('AQI表date/location示例:')
print(aqi[['date','location']].head())
print('weather表date/location示例:')
print(weather[['date','location']].head())
# 合并
df_merge = pd.merge(aqi, weather, on=['date', 'location'])
print('合并后行数:', len(df_merge))
merged = df_merge
# 气象特征7天滚动均值/最大值
for col in ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'windspeed_10m_max']:
    merged[f'{col}_roll7_mean'] = merged.groupby('location')[col].transform(lambda x: x.rolling(window=7, min_periods=1).mean().round(1))
    merged[f'{col}_roll7_max'] = merged.groupby('location')[col].transform(lambda x: x.rolling(window=7, min_periods=1).max().round(1))
# 添加衍生特征
merged['temp_range'] = (merged['temperature_2m_max'] - merged['temperature_2m_min']).round(1)
merged['is_rain'] = (merged['precipitation_sum'] > 0).astype(int)
# 其余连续特征也统一保留一位小数
for col in ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'windspeed_10m_max']:
    merged[col] = merged[col].round(1)
# 可选：保存
merged.to_csv('data/processed/merged_data.csv', index=False)

# 相关性热力图
cols = ['AQI', 'temperature_2m_max', 'temperature_2m_min', 'windspeed_10m_max', 'precipitation_sum', 'temp_range']
corr_df = pd.DataFrame(merged[cols]).corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('气象要素与AQI相关性')
plt.savefig('results/weather_aqi_corr.png')
plt.close()

# 散点图
plt.figure(figsize=(15, 4))
for i, col in enumerate(['temperature_2m_max', 'windspeed_10m_max', 'precipitation_sum']):
    plt.subplot(1, 3, i+1)
    sns.scatterplot(x=merged[col], y=merged['AQI'])
    plt.xlabel(col)
    plt.ylabel('AQI')
plt.tight_layout()
plt.savefig('results/weather_vs_aqi.png')
plt.close() 
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import numpy as np
import requests
import datetime

def load_data():
    df = pd.read_csv('data/processed/merged_data.csv')
    # 只用未来可获得的特征
    future_features = [
        'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'windspeed_10m_max',
        'month', 'season', 'week_of_year', 'day_of_year', 'dayofweek', 'is_weekend'
    ]
    # 只保留数据中实际存在的特征
    future_features = [f for f in future_features if f in df.columns]
    X = df[future_features]
    y = df['AQI']
    return X, y, df['location']

def split_train_test(X, y, locations):
    """按城市划分训练集和测试集，确保测试集包含完整城市"""
    le = LabelEncoder()
    groups = le.fit_transform(locations)
    
    # 使用GroupKFold进行划分，但只取一个fold作为测试集
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    locations_train, locations_test = locations.iloc[train_idx], locations.iloc[test_idx]
    
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"训练集城市: {locations_train.unique()}")
    print(f"测试集城市: {locations_test.unique()}")
    
    return X_train, X_test, y_train, y_test, locations_train, locations_test

def evaluate_model(model_path, X_test, y_test):
    """在测试集上评估模型"""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"测试集评估结果 - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    return model, y_pred

def analyze_model_performance_differences(X_train, X_test, y_test, locations_test, model_results):
    """深入分析模型性能差异的原因"""
    print("\n" + "="*60)
    print("模型性能差异深度分析")
    print("="*60)
    
    # 1. 城市特性分析
    print("\n1. 城市特性分析:")
    print("-" * 30)
    test_cities = locations_test.unique()
    print(f"测试集包含城市: {test_cities}")
    
    # 分析每个城市的AQI分布
    for city in test_cities:
        city_mask = locations_test == city
        city_aqi = y_test[city_mask]
        print(f"{city}: AQI均值={city_aqi.mean():.1f}, 标准差={city_aqi.std():.1f}, 范围=[{city_aqi.min():.1f}, {city_aqi.max():.1f}]")
    
    # 2. 模型在不同城市的表现
    print("\n2. 各模型在不同城市的RMSE表现:")
    print("-" * 40)
    for model_name, rmse, model, y_pred in model_results:
        print(f"\n{model_name}:")
        for city in test_cities:
            city_mask = locations_test == city
            city_y_true = y_test[city_mask]
            city_y_pred = y_pred[city_mask]
            city_rmse = mean_squared_error(city_y_true, city_y_pred) ** 0.5
            print(f"  {city}: RMSE = {city_rmse:.2f}")
    
    # 3. 特征重要性差异分析
    print("\n3. 特征重要性差异分析:")
    print("-" * 30)
    for model_name, rmse, model, y_pred in model_results:
        print(f"\n{model_name} 特征重要性:")
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            feature_names = X_test.columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            for _, row in importance_df.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 4. 预测误差分析
    print("\n4. 预测误差分析:")
    print("-" * 20)
    for model_name, rmse, model, y_pred in model_results:
        errors = y_test - y_pred
        print(f"\n{model_name}:")
        print(f"  平均误差: {errors.mean():.2f}")
        print(f"  误差标准差: {errors.std():.2f}")
        print(f"  最大正误差: {errors.max():.2f}")
        print(f"  最大负误差: {errors.min():.2f}")
        print(f"  误差>10的样本数: {(abs(errors) > 10).sum()}")
    
    # 5. 数据分布差异分析
    print("\n5. 训练集与测试集数据分布差异:")
    print("-" * 35)
    
    print("特征分布对比 (训练集 vs 测试集):")
    for feature in X_test.columns:
        train_mean = X_train[feature].mean()
        test_mean = X_test[feature].mean()
        train_std = X_train[feature].std()
        test_std = X_test[feature].std()
        print(f"  {feature}:")
        print(f"    均值: {train_mean:.2f} vs {test_mean:.2f} (差异: {abs(train_mean-test_mean):.2f})")
        print(f"    标准差: {train_std:.2f} vs {test_std:.2f} (差异: {abs(train_std-test_std):.2f})")
    
    # 6. 模型复杂度分析
    print("\n6. 模型复杂度分析:")
    print("-" * 20)
    for model_name, rmse, model, y_pred in model_results:
        print(f"\n{model_name}:")
        if hasattr(model, 'n_estimators'):
            print(f"  树的数量: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"  最大深度: {model.max_depth}")
        if hasattr(model, 'learning_rate'):
            print(f"  学习率: {model.learning_rate}")
    
    # 7. 交叉验证vs测试集性能差异解释
    print("\n7. 交叉验证vs测试集性能差异解释:")
    print("-" * 40)
    print("可能的原因:")
    print("1. 城市特异性: 测试集城市(鞍山、朝阳、葫芦岛)可能具有独特的")
    print("   气象和地理特征，与训练集城市差异较大")
    print("2. 数据量差异: 测试集仅42个样本，交叉验证使用更多数据")
    print("3. 模型过拟合: LightGBM可能在训练集上过拟合，泛化能力差")
    print("4. 特征分布偏移: 测试集的特征分布与训练集存在差异")
    print("5. 随机性: 不同的数据划分导致性能差异")
    
    return model_results

def shap_analysis(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('results/shap_summary.png')
    print('SHAP可解释性分析图已保存到 results/shap_summary.png')

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

def predict_future_aqi(model_path='results/best_model.pkl', days=7, output_path='results/future_7days_aqi_pred.csv'):
    # 城市及坐标（全省城市）
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
    model = joblib.load(model_path)
    future_features = [
        'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'windspeed_10m_max',
        'month', 'season', 'week_of_year', 'day_of_year', 'dayofweek', 'is_weekend'
    ]
    results = []
    for city, (lat, lon) in city_coords.items():
        df_weather = fetch_future_weather(city, lat, lon, days=days)
        for idx, row in df_weather.iterrows():
            features = {}
            features['temperature_2m_max'] = row['temperature_2m_max']
            features['temperature_2m_min'] = row['temperature_2m_min']
            features['precipitation_sum'] = row['precipitation_sum']
            features['windspeed_10m_max'] = row['windspeed_10m_max']
            date = pd.to_datetime(row['time'])
            # 如果date不是单个标量，取第一个元素并转为str
            if isinstance(date, (pd.DatetimeIndex, pd.Series, np.ndarray, list)):
                date = str(date[0])
            # 强制转为Timestamp
            date = pd.Timestamp(date)
            if not isinstance(date, pd.Timestamp) or pd.isnull(date) or date is pd.NaT:
                continue  # 跳过无效日期
            features['month'] = date.month
            features['season'] = date.month % 12 // 3 + 1
            features['week_of_year'] = date.isocalendar()[1]
            features['day_of_year'] = date.timetuple().tm_yday
            features['dayofweek'] = date.weekday()
            features['is_weekend'] = 1 if date.weekday() >= 5 else 0
            X_pred = pd.DataFrame([features])
            # 补齐缺失特征
            for f in future_features:
                if f not in X_pred.columns:
                    X_pred[f] = 0
            X_pred = X_pred[future_features]
            pred_aqi = model.predict(X_pred)[0]
            results.append({'location': city, 'date': date, 'AQI_pred': pred_aqi})
        print(f'{city}预测完成')
    df_pred = pd.DataFrame(results)
    df_pred['AQI_pred'] = df_pred['AQI_pred'].round(2)
    df_pred.to_csv(output_path, index=False)
    print(f'所有城市未来{days}天AQI预测已保存到 {output_path}')

def main():
    X, y, locations = load_data()
    X_train, X_test, y_train, y_test, locations_train, locations_test = split_train_test(X, y, locations)
    
    model_names = ['random_forest_model.pkl', 'xgboost_model.pkl', 'lightgbm_model.pkl']
    results = []
    for model_name in model_names:
        try:
            print(f'\n模型: {model_name}')
            model, y_pred = evaluate_model(f'results/{model_name}', X_test, y_test)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            results.append((model_name, rmse, model, y_pred))
        except FileNotFoundError:
            print(f"{model_name} 不存在，跳过。")
    
    # 选出RMSE最小的模型
    if results:
        best = min(results, key=lambda x: x[1])
        print(f"\n最优模型: {best[0]}，测试集RMSE={best[1]:.2f}")
        shap_analysis(best[2], X_test)
        # 保存最优模型为 best_model.pkl
        import joblib
        joblib.dump(best[2], 'results/best_model.pkl')
        print("最优模型已保存为 results/best_model.pkl")
        # 保存真实值和预测值
        pd.DataFrame({'真实AQI': y_test, '预测AQI': best[3]}).to_csv('results/pred_vs_true.csv', index=False)
        print("测试集真实值与预测值已保存到 results/pred_vs_true.csv")
        # 新增：预测未来7天AQI
        predict_future_aqi()
        # 新增：模型性能差异分析
        analyze_model_performance_differences(X_train, X_test, y_test, locations_test, results)
    else:
        print("没有可用的模型文件。")

if __name__ == '__main__':
    main() 
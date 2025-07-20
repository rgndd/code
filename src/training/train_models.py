import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_processed_data():
    # 读取包含气象特征的合并数据
    aqi = pd.read_csv('data/processed/merged_data.csv')
    aqi['location'] = aqi['location'].str.replace('市', '', regex=False)
    return aqi

def train_ml_models_cv(df):
    # 只用未来可获得的特征（移除所有滞后特征和滚动特征）
    future_features = [
        'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'windspeed_10m_max',
        'month', 'season', 'week_of_year', 'day_of_year', 'dayofweek', 'is_weekend'
    ]
    # 只保留数据中实际存在的特征
    future_features = [f for f in future_features if f in df.columns]
    print(f"使用的特征: {future_features}")
    
    X = df[future_features]
    y = df['AQI']
    
    # 数据验证
    print(f"数据形状: {X.shape}")
    print(f"特征数量: {len(future_features)}")
    print(f"样本数量: {len(X)}")
    print(f"特征/样本比: {len(future_features)/len(X):.3f}")
    
    # 检查数据质量
    print(f"缺失值数量: {X.isnull().sum().sum()}")
    print(f"AQI范围: {y.min():.1f} - {y.max():.1f}")
    
    # 对location编码
    le = LabelEncoder()
    groups = le.fit_transform(df['location'])
    
    # 确保特征为float
    X = X.astype(float)
    
    # 使用更少的折数，因为数据量小
    n_splits = 3
    gkf = GroupKFold(n_splits=n_splits)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=50, max_depth=3, random_state=42)
    }
    
    for name, model in models.items():
        rmses = []
        print(f'\n{name} 交叉验证:')
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            rmses.append(rmse)
            print(f'  第{fold+1}折 RMSE: {rmse:.2f} (训练集: {len(X_train)}, 测试集: {len(X_test)})')
        print(f'  平均RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}')
        
        # 用全部数据再训练并保存模型
        model.fit(X, y)
        joblib.dump(model, f'results/{name.replace(" ", "_").lower()}_model.pkl')
        
        # 输出特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = pd.Series(importances, index=future_features).sort_values(ascending=False)
            print(f'  特征重要性:\n{feature_importance}\n')
        elif hasattr(model, 'get_booster'):
            # XGBoost特征重要性
            booster = model.get_booster()
            importances = booster.get_score(importance_type='weight')
            feature_importance = pd.Series(importances).sort_values(ascending=False)
            print(f'  特征重要性:\n{feature_importance}\n')

def main():
    df = load_processed_data()
    train_ml_models_cv(df)
    print('交叉验证训练完成，模型已保存到 results/')

if __name__ == '__main__':
    main() 
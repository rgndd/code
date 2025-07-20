# 城市空气质量预测与健康风险评估系统

## 项目简介

本项目为课程作业，基于机器学习方法，融合气象与空气质量数据，实现辽宁省主要城市的空气质量（AQI）预测与可视化分析。项目流程涵盖数据采集、预处理、特征工程、模型训练、评估与可视化，帮助理解数据驱动的环境建模流程。

---

## 项目结构

```
code/
├── data/
│   ├── raw/                # 原始数据（air_quality.csv, weather_liaoning.csv）
│   └── processed/          # 处理后数据（processed_data.csv, merged_data.csv）
├── docs/                   # 文档模板
├── notebooks/              # Jupyter演示
├── results/                # 结果输出（模型、图表、预测结果等）
├── src/
│   ├── data/               # 数据处理与采集脚本
│   ├── evaluation/         # 模型评估与未来预测
│   ├── training/           # 训练脚本
│   └── visualization/      # 可视化脚本
├── requirements.txt        # 依赖包
└── README.md
```

---

## 环境依赖

- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- joblib
- requests
- shap

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 快速开始

**请在项目根目录下依次运行以下脚本：**

1. **数据处理与特征工程**
   ```bash
   python src/data/data_processing.py
   ```
   - 处理原始空气质量和气象数据，生成 `data/processed/processed_data.csv` 和 `data/processed/merged_data.csv`。

2. **气象数据采集与合并**
   ```bash
   python src/data/fetch_weather_liaoning.py
   ```
   - 自动抓取辽宁省各城市气象数据，并与AQI数据合并，生成特征丰富的训练数据。

3. **模型训练**
   ```bash
   python src/training/train_models.py
   ```
   - 训练随机森林、XGBoost、LightGBM三种模型，交叉验证并保存模型。

4. **模型评估与未来预测**
   ```bash
   python src/evaluation/evaluate.py
   ```
   - 评估模型性能，输出RMSE等指标，保存最佳模型，并预测未来7天AQI。

5. **结果可视化**
   ```bash
   python src/visualization/visualize.py
   ```
   - 生成预测对比、分布、特征重要性、未来AQI等多种可视化图表，结果保存在 `results/` 目录。

---

## 主要功能

- **数据采集与预处理**：自动抓取气象数据，清洗并合并空气质量数据。
- **特征工程**：生成时序、气象、滞后等多种特征。
- **多模型训练与对比**：支持随机森林、XGBoost、LightGBM等主流机器学习模型。
- **模型评估与可解释性分析**：输出RMSE、MAE、R2等指标，支持SHAP可解释性分析。
- **未来AQI预测**：基于历史与气象数据，预测未来7天各城市AQI。
- **可视化分析**：自动生成多种分析图表，便于结果展示与理解。

---

## 结果示例

- 预测值与真实值对比图：`results/pred_vs_true.png`
- 预测分布图：`results/pred_distribution.png`
- 特征重要性图：`results/feature_importance.png`
- SHAP可解释性图：`results/shap_summary.png`
- 未来7天AQI预测图：`results/future_aqi_forecast.png`
- 所有城市未来7天AQI预测图：`results/future_7days_aqi_pred.png`

---

## 注意事项

- **请确保在项目根目录下运行所有脚本。**
- 若需重新采集气象数据，请保证网络畅通。
- 若部分依赖未安装，请先运行 `pip install -r requirements.txt`。

---

## 课程作业说明

本项目为课程作业，所有代码和数据仅用于学习与交流。欢迎同学参考和改进。

---

## 联系方式

如有问题，请通过课程平台或邮件联系作者。 
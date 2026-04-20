import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from catboost import CatBoostClassifier

# 1. 创建结果文件夹
output_dir = "catboost_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 加载数据
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print("数据加载成功！")
except FileNotFoundError:
    print("错误：未找到 train.csv 或 test.csv，请检查路径。")
    exit()

targets = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
features = [col for col in df_train.columns if col not in targets and col != 'id']

X = df_train[features]
y = df_train[targets]
X_test_final = df_test[features]

# 3. 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_final_scaled = scaler.transform(X_test_final)

# 划分训练集和验证集 (80/20)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. 训练 CatBoost 模型
print("正在训练 CatBoost (基于对称树结构的最新集成算法)...")
# 核心修复：针对多标签任务，通过 MultiOutputClassifier 包装，底层使用 Logloss + AUC
cat = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.0178,
    depth=4,
    l2_leaf_reg=2.07,
    loss_function='Logloss',  # 修复兼容性：每个子分类器进行二分类优化
    eval_metric='AUC',        # 评价指标设为 AUC
    random_seed=42,
    verbose=100               # 每100代打印一次 log
)

# 使用多标签包装器，这会为 7 种缺陷各训练一个独立的 CatBoost 模型
model = MultiOutputClassifier(cat)
model.fit(X_train, y_train)

# 5. 验证与指标计算
print("\n正在计算验证集评估指标...")
y_pred_val = model.predict(X_val)
# 提取概率：predict_proba 返回列表，需重组为 (样本数, 标签数) 的矩阵
y_prob_val = np.array(model.predict_proba(X_val))[:, :, 1].T

mse = mean_squared_error(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)
auc = roc_auc_score(y_val, y_prob_val, multi_class='ovr')

# 6. 保存指标到 TXT (用于报告定量分析)
metrics_path = os.path.join(output_dir, "evaluation_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("=== CatBoost Training Results ===\n")
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"R-squared (R2): {r2:.4f}\n")
    f.write(f"ROC-AUC Score: {auc:.4f}\n")
    f.write("\nNote: MSE/MAE/R2 are calculated on binary predictions (0/1).")

# 7. 生成正式提交文件 (Kaggle 推理)
print("正在对 test.csv 进行推理...")
y_prob_test = np.array(model.predict_proba(X_test_final_scaled))[:, :, 1].T
submission = pd.DataFrame(y_prob_test, columns=targets)
submission.insert(0, 'id', df_test['id'])
submission.to_csv('submission_catboost.csv', index=False)

# 8. 保存模型和标准化器
joblib.dump(model, os.path.join(output_dir, "cat_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "cat_scaler.pkl"))

print(f"\n训练完成！CatBoost 本地验证集 ROC-AUC: {auc:.4f}")
print(f"结果已保存至文件夹: {output_dir}")
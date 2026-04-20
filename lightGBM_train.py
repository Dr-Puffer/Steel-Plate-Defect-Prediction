import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
import joblib
import warnings

# 忽略不影响结果的细微警告
warnings.filterwarnings('ignore', category=UserWarning)

# 1. 创建结果文件夹
output_dir = "lightgbm_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 加载数据
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')  # 加载用于提交的测试集
    print("数据加载成功！")
except FileNotFoundError:
    print("错误：未找到 train.csv 或 test.csv 文件，请检查路径。")
    exit()

# 定义目标列和特征列
targets = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
features = [col for col in df_train.columns if col not in targets and col != 'id']

X = df_train[features]
y = df_train[targets]
X_test_final = df_test[features]

# 3. 数据预处理
scaler = StandardScaler()
# 划分本地验证集 (80/20)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化：保持 DataFrame 格式以保留列名
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=features)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_raw), columns=features)
X_test_final_scaled = pd.DataFrame(scaler.transform(X_test_final), columns=features)

# 4. 使用 Optuna 搜索到的最优参数训练模型
print("正在训练 LightGBM（优化版）...")
lgbm = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01011599159147172,
    num_leaves=33,
    max_depth=9,
    min_data_in_leaf=49,
    feature_fraction=0.24180823614705715,
    bagging_fraction=0.7,
    bagging_freq=5,
    random_state=42,
    verbose=-1,
    lambda_l2=0.017225865694717367
)

# 使用多标签包装器
model = MultiOutputClassifier(lgbm)
model.fit(X_train_scaled, y_train)

# 5. 本地验证集评估
y_pred_val = model.predict(X_val_scaled)
# 提取概率用于计算 ROC-AUC
y_prob_val = np.array(model.predict_proba(X_val_scaled))[:, :, 1].T

auc = roc_auc_score(y_val, y_prob_val, multi_class='ovr')
print(f"本地验证集 ROC-AUC 得分: {auc:.4f}")

# 6. 保存指标到 TXT
metrics_path = os.path.join(output_dir, "evaluation_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("=== LightGBM Training Results ===\n")
    f.write(f"ROC-AUC Score: {auc:.4f}\n")

# 7. 生成正式提交文件 (针对 Kaggle test.csv)
print("正在对 test.csv 进行推理并生成提交文件...")
# 必须使用 predict_proba 预测 0-1 之间的概率值
y_prob_test = np.array(model.predict_proba(X_test_final_scaled))[:, :, 1].T

submission = pd.DataFrame(y_prob_test, columns=targets)
submission.insert(0, 'id', df_test['id'])
submission.to_csv('lgbm_submission.csv', index=False)
print("提交文件已生成: lgbm_submission.csv")

# 8. 保存模型和相关元数据
joblib.dump(model, os.path.join(output_dir, "lgbm_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
joblib.dump(features, os.path.join(output_dir, "feature_names.pkl"))

print(f"所有流程完成！结果已保存至: {output_dir}")
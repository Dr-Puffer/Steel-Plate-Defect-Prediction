import pandas as pd
import numpy as np
import optuna
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)

# 1. 加载数据 (包含训练集和特定的测试集)
try:
    df_train = pd.read_csv('train.csv')
    df_test_final = pd.read_csv('test.csv')  # 你提到的特定 test 文件
    print("数据加载成功！")
except FileNotFoundError:
    print("错误：未找到 train.csv 或 test.csv。")
    exit()

targets = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
features = [col for col in df_train.columns if col not in targets and col != 'id']

X = df_train[features]
y = df_train[targets]
X_test_final = df_test_final[features]

# 2. 数据预处理 (保持 DataFrame 格式以保留特征名)
scaler = StandardScaler()
# 划分验证集用于探索参数
X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=features)
X_val = pd.DataFrame(scaler.transform(X_val_raw), columns=features)
X_test_final_scaled = pd.DataFrame(scaler.transform(X_test_final), columns=features)


# 3. 定义 Optuna 探索函数
def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 10.0, log=True),
        'random_state': 42
    }

    lgbm = lgb.LGBMClassifier(**param, n_estimators=500)
    model = MultiOutputClassifier(lgbm)
    model.fit(X_train, y_train)

    # 在验证集上评估
    y_prob_val = np.array(model.predict_proba(X_val))[:, :, 1].T
    return roc_auc_score(y_val, y_prob_val, multi_class='ovr')


# 4. 执行探索
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)

print(f"\n最佳验证集 AUC: {study.best_value:.5f}")
print("最佳参数:", study.best_params)

# 5. 使用最佳参数对特定的 test.csv 进行最终测试
print("\n正在使用最优参数对 test.csv 进行推理...")
best_model = MultiOutputClassifier(lgb.LGBMClassifier(**study.best_params, n_estimators=1000, verbosity=-1))
best_model.fit(X_train, y_train)  # 也可以用全部 X, y 重新训练

# 生成预测结果
final_probs = np.array(best_model.predict_proba(X_test_final_scaled))[:, :, 1].T

# 6. 保存为 Kaggle 提交格式
submission = pd.DataFrame(final_probs, columns=targets)
submission.insert(0, 'id', df_test_final['id'])
submission.to_csv("lgbm_submission.csv", index=False)
print("最终预测结果已保存至 lgbm_submission.csv")
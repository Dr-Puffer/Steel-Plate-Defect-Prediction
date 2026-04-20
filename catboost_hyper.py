import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import os

# 1. 加载与预处理
df_train = pd.read_csv('train.csv')
targets = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
features = [col for col in df_train.columns if col not in targets and col != 'id']

X = df_train[features]
y = df_train[targets]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 2. 定义目标函数
def objective(trial):
    # 定义探索范围 (根据 Records.txt 的发现，重点探索浅层树)
    param = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 1, 4),  # 重点关注 1-4 层
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False
    }

    cat = CatBoostClassifier(**param)
    model = MultiOutputClassifier(cat)
    model.fit(X_train, y_train)

    # 计算验证集 AUC
    y_prob_val = np.array(model.predict_proba(X_val))[:, :, 1].T
    auc = roc_auc_score(y_val, y_prob_val, multi_class='ovr')

    return auc


# 3. 执行探索
print("开始自动超参数探索...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)  # 运行30次实验

# 4. 输出结果
print("\n=== 探索完成 ===")
print(f"最优 ROC-AUC: {study.best_value:.4f}")
print("最优参数组合:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# 保存搜索结果供报告使用
if not os.path.exists("analysis_results"): os.makedirs("analysis_results")
study.trials_dataframe().to_csv("analysis_results/hyperparameter_trials.csv", index=False)
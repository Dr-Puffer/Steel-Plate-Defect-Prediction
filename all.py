import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

# 导入算法
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 环境配置与数据加载
warnings.filterwarnings('ignore')
targets = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    sample_sub = pd.read_csv('sample_submission.csv')
    print("数据加载成功！")
except FileNotFoundError:
    print("错误：未找到 CSV 数据文件。")
    exit()

# 2. 统一预处理流程
features = [col for col in df_train.columns if col not in targets and col != 'id']
X = df_train[features]
y = df_train[targets]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_final_scaled = pd.DataFrame(scaler.transform(df_test[features]), columns=features)

# 划分本地验证集 (80/20)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_df = pd.DataFrame(X_train, columns=features)
X_val_df = pd.DataFrame(X_val, columns=features)

# 3. 定义算法全家桶
models_config = {
    "Logistic Regression": {
        "model": MultiOutputClassifier(LogisticRegression(max_iter=1000)),
        "output_dir": "logistic_results"
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42),
        "output_dir": "rf_results"
    },
    "XGBoost": {
        "model": MultiOutputClassifier(
            XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, tree_method='hist', random_state=42)),
        "output_dir": "xgboost_results"
    },
    "CatBoost": {
        "model": MultiOutputClassifier(
            CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, verbose=0, random_seed=42)),
        "output_dir": "catboost_results"
    },
    "LightGBM": {
        "model": MultiOutputClassifier(
            LGBMClassifier(n_estimators=1000, learning_rate=0.02, num_leaves=31, verbose=-1, random_state=42)),
        "output_dir": "lightgbm_results"
    }
}

# 4. 循环训练与评估
perf_summary = []

for name, config in models_config.items():
    print(f"\n>>> 正在处理: {name}")
    model = config["model"]
    out_dir = config["output_dir"]
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    model.fit(X_train_df, y_train)

    if name == "Random Forest":
        y_prob_list = model.predict_proba(X_val_df)
        y_prob_val = np.transpose([p[:, 1] for p in y_prob_list])
    else:
        y_prob_val = np.array(model.predict_proba(X_val_df))[:, :, 1].T

    auc = roc_auc_score(y_val, y_prob_val, multi_class='ovr')
    perf_summary.append({"Algorithm": name, "ROC-AUC": auc})

    # 保存结果
    sub_file = sample_sub.copy()
    if name == "Random Forest":
        y_prob_test_list = model.predict_proba(X_test_final_scaled)
        y_prob_test = np.transpose([p[:, 1] for p in y_prob_test_list])
    else:
        y_prob_test = np.array(model.predict_proba(X_test_final_scaled))[:, :, 1].T

    sub_file[targets] = y_prob_test
    sub_file.to_csv(f'submission_{name.lower().replace(" ", "_")}.csv', index=False)

# 5. 绘制 20 号字体的对比图
print("\n--- 正在生成大字体对比图 ---")
df_perf = pd.DataFrame(perf_summary).sort_values(by="ROC-AUC", ascending=True)

# 设置全局字体大小为 20
plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(16, 10))  # 调大画布以适应大字体
sns.set_style("whitegrid")

# 绘制条形图
ax = sns.barplot(x="ROC-AUC", y="Algorithm", data=df_perf, palette="Blues_d")

# 调整细节字体
plt.title('Algorithm Comparison: Validation ROC-AUC', fontsize=24, pad=20)  # 标题可以稍微再大点
plt.xlabel('ROC-AUC Score', fontsize=20)
plt.ylabel('Algorithm', fontsize=20)

# 设置 X 轴范围
plt.xlim(max(0.8, df_perf["ROC-AUC"].min() - 0.01), min(1.0, df_perf["ROC-AUC"].max() + 0.01))

# 在柱状图末端添加 20 号字体的数值
for i, v in enumerate(df_perf["ROC-AUC"]):
    ax.text(v + 0.0005, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=20)

plt.tight_layout()
plt.savefig('comparison_results_large_font.png', dpi=300)
plt.show()

print("\n任务完成！大字体图表已保存。")
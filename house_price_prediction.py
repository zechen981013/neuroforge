"""
Ames 房价预测（Kaggle House Prices 竞赛流程复现）
数据: data/house_prices.csv (1460 行, 79 特征 + SalePrice)
流程: 加载 -> 探索 -> 预处理 -> 建模 -> 评估 -> 特征重要性
输出: data/predictions.csv (预测结果) | results.txt (报告) | prediction_results.png (可视化)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面环境绘图
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 中文字体支持 (macOS)
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 加载数据 ----------
df = pd.read_csv("data/house_prices.csv")
print("=" * 60)
print(f"数据规模: {df.shape[0]} 行 x {df.shape[1]} 列")
print("缺失值最多的列:", df.isnull().mean().sort_values(ascending=False).head(3).round(3).to_dict())

y = df["SalePrice"]
X_all = df.drop(columns=["SalePrice"])  # 保留 Id 以便回写预测

# ---------- 2. 目标变量处理: 对数变换 ----------
y_log = np.log1p(y)
print("SalePrice 右偏度(>0 右偏):", round(float(y.skew()), 2), "-> log 后偏度:", round(float(y_log.skew()), 2))

# ---------- 3. 预处理流水线 ----------
X = X_all.drop(columns=["Id"])
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
print(f"数值特征 {len(num_cols)} 个, 分类特征 {len(cat_cols)} 个")

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),          # 缺失值用中位数填充(稳健)
    ("scaler", StandardScaler()),                            # 标准化 (z-score)
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),   # 缺失值用众数填充
    ("onehot", OneHotEncoder(handle_unknown="ignore")),      # 独热编码
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

# ---------- 4. 训练/验证划分 ----------
X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)
_, ids_val = train_test_split(X_all["Id"], test_size=0.2, random_state=42)

# ---------- 5. 模型对比 ----------
models = {
    "随机森林 RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "梯度提升 GradientBoosting": GradientBoostingRegressor(random_state=42),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred_log = pipe.predict(X_val)
    rmse_log = float(np.sqrt(mean_squared_error(y_val, pred_log)))
    pred_price = np.expm1(pred_log)   # 还原成真实价格
    true_price = np.expm1(y_val)
    rmse_price = float(np.sqrt(mean_squared_error(true_price, pred_price)))
    results[name] = (rmse_log, rmse_price)
    print(f"{name:<20} | log-RMSE: {rmse_log:.4f} | 原价格RMSE: ${rmse_price:,.0f}")

# ---------- 6. 最优模型: 重新训练、保存预测与报告 ----------
best_name = min(results, key=lambda k: results[k][0])
print(f"\n最优模型: {best_name}")

pipe = Pipeline([("prep", preprocessor), ("model", models[best_name])])
pipe.fit(X_train, y_train)
model = pipe.named_steps["model"]

# 6.1 保存预测结果 CSV (验证集的预测价格)
pred_log = pipe.predict(X_val)
pred_price = np.expm1(pred_log)
true_price = np.expm1(y_val)
predictions = pd.DataFrame({
    "Id": ids_val.values,
    "真实价格": true_price.round(0).astype(int),
    "预测价格": pred_price.round(0).astype(int),
    "误差": (true_price - pred_price).round(0).astype(int),
    "误差比例": ((true_price - pred_price) / true_price * 100).round(2),
})
predictions = predictions.sort_values("Id").reset_index(drop=True)
predictions.to_csv("data/predictions.csv", index=False)
print("预测结果已保存 -> data/predictions.csv")

# 6.2 保存结果报告 txt
onehot = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
cat_names = onehot.get_feature_names_out(cat_cols)
all_names = np.array(num_cols + list(cat_names))
imp = pd.Series(model.feature_importances_, index=all_names).sort_values(ascending=False)

with open("results.txt", "w", encoding="utf-8") as f:
    f.write("Ames 房价预测结果报告\n")
    f.write("=" * 50 + "\n")
    f.write(f"数据规模: {df.shape[0]} 行 x {df.shape[1]} 列\n")
    f.write(f"SalePrice 偏度: {float(y.skew()):.2f} -> log 后: {float(y_log.skew()):.2f}\n\n")
    for name, (rmse_log, rmse_price) in results.items():
        f.write(f"{name:<20} | log-RMSE: {rmse_log:.4f} | 原价格RMSE: ${rmse_price:,.0f}\n")
    f.write(f"\n最优模型: {best_name}\n\nTop 10 重要特征:\n")
    f.write(imp.head(10).round(4).to_string())
print("结果报告已保存 -> results.txt")

# ---------- 7. 可视化: 预测 vs 真实 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左图: 预测价格 vs 真实价格 散点图
ax = axes[0]
ax.scatter(true_price, pred_price, alpha=0.5, s=20, color="#4C72B0")
lim = [0, max(true_price.max(), pred_price.max()) * 1.05]
ax.plot(lim, lim, "--", color="red", linewidth=1.5, label="完美预测 (y=x)")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("真实价格 ($)")
ax.set_ylabel("预测价格 ($)")
ax.set_title(f"{best_name}\n验证集: log-RMSE = {results[best_name][0]:.4f}")
ax.legend()

# 右图: 残差分布 (预测 - 真实)
ax = axes[1]
residual = pred_price - true_price
ax.scatter(true_price, residual, alpha=0.5, s=20, color="#DD8452")
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("真实价格 ($)")
ax.set_ylabel("残差 预测-真实 ($)")
ax.set_title(f"残差图 (平均误差 ${residual.mean():,.0f})\n均价 ${true_price.mean():,.0f}")

plt.tight_layout()
plt.savefig("prediction_results.png", dpi=150, bbox_inches="tight")
print("可视化已保存 -> prediction_results.png")

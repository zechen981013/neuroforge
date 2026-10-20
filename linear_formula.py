"""用线性回归拟合 log(房价)，输出可写的数学公式"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/house_prices.csv")
y_log = np.log1p(df["SalePrice"])
X = df.drop(columns=["Id", "SalePrice"])
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
Xp = prep.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(Xp, y_log, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
pred_log = lr.predict(X_val)
print("线性回归 log-RMSE: %.4f | R2: %.4f" % (np.sqrt(mean_squared_error(y_val, pred_log)), r2_score(y_val, pred_log)))

oh = prep.named_transformers_["cat"].named_steps["oh"]
names = np.array(num_cols + list(oh.get_feature_names_out(cat_cols)))
coef = pd.Series(lr.coef_, index=names)
print("截距 b0 = %.4f" % lr.intercept_)
print("\n数值特征系数 (标准化后, z-score 每 +1 的影响):")
print(coef[num_cols].sort_values(ascending=False).head(10).round(4).to_string())
print("\n影响最大的正系数 Top 8 (分类):")
print(coef.sort_values(ascending=False).head(8).round(4).to_string())
print("\n影响最大的负系数 Top 5:")
print(coef.sort_values().head(5).round(4).to_string())

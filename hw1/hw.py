# 가상환경 설정
# conda create --name myenv38 python=3.8
# conda activate myenv38
# conda install pandas numpy xgboost scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
file_path = '/Users/t2023-m0093/Desktop/homework/hw1/housingdata.csv'
housing_data = pd.read_csv(file_path)

# 데이터 확인
print(housing_data.head())
print(housing_data.info())
print(housing_data.describe())

# 결측치 확인 및 제거
print(housing_data.isnull().sum())
housing_data = housing_data.dropna()

# 이상치 탐지 및 제거
plt.figure(figsize=(10, 6))
sns.boxplot(data=housing_data, x='RM')
plt.title('RM Boxplot')
plt.show()

# RM 열의 이상치 제거 (임의의 기준 설정)
Q1 = housing_data['RM'].quantile(0.25)
Q3 = housing_data['RM'].quantile(0.75)
IQR = Q3 - Q1
rm_outliers = housing_data[(housing_data['RM'] < (Q1 - 1.5 * IQR)) | (housing_data['RM'] > (Q3 + 1.5 * IQR))]
housing_data = housing_data.drop(rm_outliers.index)

# 특징과 타겟 변수 분리
X = housing_data[['RM', 'LSTAT', 'PTRATIO', 'AGE']]
y = housing_data['MEDV']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 초기화
lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor(random_state=42)
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# 모델 학습
lin_reg.fit(X_train_scaled, y_train)
tree_reg.fit(X_train_scaled, y_train)
forest_reg.fit(X_train_scaled, y_train)

# 예측
y_train_pred_lin = lin_reg.predict(X_train_scaled)
y_test_pred_lin = lin_reg.predict(X_test_scaled)

y_train_pred_tree = tree_reg.predict(X_train_scaled)
y_test_pred_tree = tree_reg.predict(X_test_scaled)

y_train_pred_forest = forest_reg.predict(X_train_scaled)
y_test_pred_forest = forest_reg.predict(X_test_scaled)

# 성능 평가
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

metrics = {
    'Linear Regression': evaluate_model(y_test, y_test_pred_lin),
    'Decision Tree': evaluate_model(y_test, y_test_pred_tree),
    'Random Forest': evaluate_model(y_test, y_test_pred_forest)
}

# 성능 지표 출력
for model_name, (mae, mse, r2) in metrics.items():
    print(f'{model_name} - MAE: {mae}, MSE: {mse}, R²: {r2}')

# 성능 지표를 데이터프레임으로 정리
metrics_df = pd.DataFrame(metrics, index=['MAE', 'MSE', 'R²'])

# 시각화
plt.figure(figsize=(12, 6))
metrics_df.T.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.show()

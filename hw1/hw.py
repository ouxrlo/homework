# 가상환경 설정
# conda create --name myenv38 python=3.8
# conda activate myenv38
# conda install pandas numpy xgboost scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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

# 데이터 확인 (상위 5개 행과 기본 정보)
print(housing_data.head())
print(housing_data.info())
print(housing_data.describe())

# 결측치 확인 및 제거
print(housing_data.isnull().sum())  # 결측치 개수 확인
housing_data = housing_data.dropna()  # 결측치가 있는 행 제거

# 이상치 탐지 및 제거 (RM 열을 기준으로)
plt.figure(figsize=(10, 6))
sns.boxplot(data=housing_data, x='RM')
plt.title('RM Boxplot')
plt.show()

# IQR 방법을 사용하여 RM 열의 이상치 제거
Q1 = housing_data['RM'].quantile(0.25)
Q3 = housing_data['RM'].quantile(0.75)
IQR = Q3 - Q1
rm_outliers = housing_data[(housing_data['RM'] < (Q1 - 1.5 * IQR)) | (housing_data['RM'] > (Q3 + 1.5 * IQR))]
housing_data = housing_data.drop(rm_outliers.index)

# 특성 엔지니어링: 새로운 특성 추가 (RM과 LSTAT의 곱, AGE의 제곱)
housing_data['RM_LSTAT'] = housing_data['RM'] * housing_data['LSTAT']
housing_data['AGE_SQ'] = housing_data['AGE'] ** 2

# 목표 변수 MEDV에 로그 변환 적용 (정규화 효과)
housing_data['MEDV'] = np.log1p(housing_data['MEDV'])

# 특징(X)과 목표 변수(y) 분리
X = housing_data[['RM', 'LSTAT', 'PTRATIO', 'AGE', 'RM_LSTAT', 'AGE_SQ']]
y = housing_data['MEDV']

# 데이터 분할: 훈련 데이터(80%)와 테스트 데이터(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링 (StandardScaler 사용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 초기화
lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor(random_state=42)
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# 하이퍼파라미터 튜닝을 위한 그리드 서치 설정 (Random Forest 모델)
param_grid = {
    'n_estimators': [100, 200, 300],  # 트리의 개수
    'max_depth': [10, 20, None],  # 트리의 최대 깊이
    'min_samples_split': [2, 5, 10],  # 분할을 위한 최소 샘플 수
    'min_samples_leaf': [1, 2, 4]  # 리프 노드에서 최소 샘플 수
}

# 그리드 서치로 모델 최적화 (교차 검증 사용)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# 최적의 파라미터 출력
print("Best Parameters:", grid_search.best_params_)

# 최적화된 모델로 예측
best_forest_reg = grid_search.best_estimator_

# 예측
y_train_pred_forest = best_forest_reg.predict(X_train_scaled)
y_test_pred_forest = best_forest_reg.predict(X_test_scaled)

# 성능 평가 함수 정의
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# 성능 평가
metrics = {
    'Random Forest (Optimized)': evaluate_model(y_test, y_test_pred_forest)
}

# 성능 지표 출력
for model_name, (mae, mse, r2) in metrics.items():
    print(f'{model_name} - MAE: {mae}, MSE: {mse}, R²: {r2}')

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

# 성능 비교 시각화
plt.figure(figsize=(12, 6))
metrics_df.T.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.show()

# Random Forest 모델에서 특성 중요도 추출
feature_importances = best_forest_reg.feature_importances_

# 중요도를 데이터프레임으로 변환
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# 특성 중요도 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance - Random Forest')
plt.show()

# 예측값과 실제값 비교 시각화 (Random Forest)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_forest, color='blue', label='Predicted', alpha=0.5)
plt.scatter(y_test, y_test, color='red', label='Actual', alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices - Random Forest')
plt.legend()
plt.show()



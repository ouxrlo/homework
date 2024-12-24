
# 가상환경 설정
# conda create --name myenv38 python=3.8
# conda activate myenv38
# conda install pandas numpy xgboost scikit-learn matplotlib seaborn


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 주택 데이터 로드
file_path = '/Users/t2023-m0093/Desktop/homework/hw1/housingdata.csv'
housing_data = pd.read_csv(file_path)

# 주어진 데이터셋에는 날짜 정보가 없으므로, 임의로 'YEAR'와 'SEASON'을 추가합니다.
# 예시로 'YEAR'를 2024년으로 설정하고, 'SEASON'을 월(MONTH) 기준으로 분류합니다.
# 실제 데이터에는 연도 정보가 없다면 외부 경제 데이터를 활용하거나 월 기준으로 가정할 수 있습니다.

# 예시로 연도 컬럼을 추가 (임의로 2024년으로 설정)
housing_data['YEAR'] = 2024  # 실제 연도에 맞게 수정 필요
housing_data['MONTH'] = 1  # 월 정보가 없다면 임의로 1월로 설정

# 계절성 변수 추가 (봄, 여름, 가을, 겨울)
housing_data['SEASON'] = housing_data['MONTH'].apply(lambda x: 'Winter' if x in [12, 1, 2] 
                                                     else ('Spring' if x in [3, 4, 5] 
                                                           else ('Summer' if x in [6, 7, 8] 
                                                                 else 'Fall')))

# 특성(X)와 목표 변수(y) 분리
X = housing_data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'YEAR', 'SEASON']]
y = housing_data['MEDV']

# 범주형 변수(계절)를 더미 변수로 변환
X = pd.get_dummies(X, columns=['SEASON'], drop_first=True)

# 데이터 분할: 훈련 데이터(90%)와 테스트 데이터(10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습 (Random Forest 모델 사용)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 예측
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# 성능 평가 함수 정의
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# 성능 평가
metrics = evaluate_model(y_test, y_test_pred)
print(f"MAE: {metrics[0]}, MSE: {metrics[1]}, R²: {metrics[2]}")

# 예측값과 실제값 비교 시각화
plt.scatter(y_test, y_test_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()















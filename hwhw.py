import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from prophet import Prophet

# 데이터 로딩 
file_path = '/Users/t2023-m0093/Desktop/hw_2/Mall_Customers.csv'
df = pd.read_csv(file_path)

# 데이터 탐색: 처음 5개의 행을 출력하여 데이터 구조 확인
print(df.head())
print(df.describe())

# 결측치 처리: SimpleImputer를 사용하여 결측치를 처리
imputer = SimpleImputer(strategy='median')  # 결측치를 중간값(median)으로 대체
df_imputed = imputer.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = df_imputed  # 대체된 값으로 데이터 업데이트

# 스케일링 : 데이터 표준화를 통해 각 특성의 평균과 분산을 0, 1로 변환
scaler = StandardScaler()  # 표준화 객체 생성
scaled_data = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])  # 데이터 표준화
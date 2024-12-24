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

# K-means 클러스터링: 데이터를 5개의 군집으로 분할
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=500, n_init=20, random_state=0)
y_kmeans = kmeans.fit_predict(scaled_data)

# 계층적 군집화 : 데이터를 4개의 군집으로 분할
agg_clustering = AgglomerativeClustering(n_clusters=4)
y_agg = agg_clustering.fit_predict(scaled_data)

# DBSCAN 클러스터링: 밀도 기반 군집화 방법
dbscan = DBSCAN(eps=0.8, min_samples=5)
y_dbscan = dbscan.fit_predict(scaled_data)

# 엘보우 방법을 통해 최적의 K값 찾기
wcss = []  # WCSS (Within-Cluster Sum of Squares)를 저장할 리스트
for i in range(1, 11):  # 1부터 10까지 K값에 대해 반복
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=20, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)  # 각 K값에 대한 WCSS 계산
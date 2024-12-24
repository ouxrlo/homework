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
file_path = '/Users/t2023-m0093/Desktop/homework/hw2/Mall_Customers.csv'

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

    # 엘보우 그래프 시각화: WCSS를 통해 최적의 K값을 시각적으로 찾기
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# 실루엣 점수를 통한 클러스터 수 평가
silhouette_scores = []  # 실루엣 점수를 저장할 리스트
for i in range(2, 11):  # 2부터 10까지 클러스터 수를 평가
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=20, random_state=0)
    y_kmeans = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, y_kmeans)  # 실루엣 점수 계산
    silhouette_scores.append(silhouette_avg)

# 실루엣 점수 그래프 시각화: 클러스터 수에 따른 실루엣 점수를 시각적으로 확인
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Scores for Different K')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# PCA로 차원 축소 후 시각화 (2D): 데이터의 차원을 2차원으로 축소하여 군집화 결과를 시각화
pca = PCA(n_components=2)  # PCA 객체 생성 (2차원으로 축소)
principalComponents = pca.fit_transform(scaled_data)  # 차원 축소 수행

# K-means 결과 시각화: 각 군집을 다른 색으로 표시
plt.figure(figsize=(8,6))
plt.scatter(principalComponents[y_kmeans == 0, 0], principalComponents[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(principalComponents[y_kmeans == 1, 0], principalComponents[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(principalComponents[y_kmeans == 2, 0], principalComponents[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(principalComponents[y_kmeans == 3, 0], principalComponents[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(principalComponents[y_kmeans == 4, 0], principalComponents[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('K-means Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# 계층적 군집화 결과 시각화
plt.figure(figsize=(8,6))
plt.scatter(principalComponents[y_agg == 0, 0], principalComponents[y_agg == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(principalComponents[y_agg == 1, 0], principalComponents[y_agg == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(principalComponents[y_agg == 2, 0], principalComponents[y_agg == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(principalComponents[y_agg == 3, 0], principalComponents[y_agg == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.title('Agglomerative Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# DBSCAN 클러스터링 결과 시각화
plt.figure(figsize=(8,6))
plt.scatter(principalComponents[y_dbscan == -1, 0], principalComponents[y_dbscan == -1, 1], s=100, c='grey', label='Noise')
plt.scatter(principalComponents[y_dbscan == 0, 0], principalComponents[y_dbscan == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(principalComponents[y_dbscan == 1, 0], principalComponents[y_dbscan == 1, 1], s=100, c='blue', label='Cluster 2')
plt.title('DBSCAN Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# GMM 클러스터링 : 데이터에 GMM을 적용
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0)
y_gmm = gmm.fit_predict(scaled_data)

# 실루엣 점수를 사용하여 클러스터링 성능 평가
silhouette_kmeans = silhouette_score(scaled_data, y_kmeans)
silhouette_dbscan = silhouette_score(scaled_data, y_dbscan) if len(set(y_dbscan)) > 1 else -1
silhouette_gmm = silhouette_score(scaled_data, y_gmm)

# 결과 출력
print(f"K-means 실루엣 점수: {silhouette_kmeans}")
print(f"DBSCAN 실루엣 점수: {silhouette_dbscan}")
print(f"GMM 실루엣 점수: {silhouette_gmm}")

# 시계열 분석 추가 (Prophet 모델)
df['Year'] = 2020  # 모든 데이터를 2020년으로 설정 (예시)
df_monthly = df.groupby('Year').agg({'Spending Score (1-100)': 'mean'}).reset_index()

# 결측값 처리 (필요 시): Prophet 모델은 결측값을 처리해야 하므로 결측값을 제거
df_monthly = df_monthly.dropna()  # 결측값이 있는 행 제거

# 데이터가 충분한지 확인 (2개 이상의 유효한 행이 있어야 모델을 학습할 수 있음)
if df_monthly.shape[0] < 2:
    print("데이터에 충분한 유효한 행이 없습니다.")
else:
    # Prophet에 맞게 데이터 포맷 변경
    df_prophet = df_monthly.rename(columns={'Year': 'ds', 'Spending Score (1-100)': 'y'})

    # Prophet 모델 생성 및 학습
    model = Prophet(yearly_seasonality=True)  # 연도별 계절성 추가
    model.fit(df_prophet)

    # 미래 데이터 예측 (2021년부터 2025년까지)
    future = model.make_future_dataframe(df_prophet, periods=5, freq='Y')

    # 예측
    forecast = model.predict(future)

    # 예측 결과 시각화
    model.plot(forecast)
    plt.title('Spending Score Prediction Over Time')
    plt.xlabel('Year')
    plt.ylabel('Spending Score (1-100)')
    plt.show()

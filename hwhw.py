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


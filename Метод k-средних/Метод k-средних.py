
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA

# Читаем файл,убираем текстовые данные
dt = pd.read_csv('data.csv')
dt = dt.drop(dt.columns[[1, 2, 3,10,11,12,13,20,21,22,23,24,25,27,28,31,32]], axis=1)
labels = dt.to_numpy('int')[:,-1]


# Добавляем нули в пропущенные строки
dt = dt.fillna(0)
# Проверка на отсутствующие данные
x = dt.values[:, :]


dtset = preprocessing.minmax_scale(x)
pca = PCA(n_components=5)
pca_dtset = pca.fit(dtset).transform(dtset)


y=sum(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_)
print("pca.explained_variance_ratio_ is ",y)
# print(pca.singular_values_)

plt.scatter(pca_dtset[:,0],pca_dtset[:,1],c=labels,cmap='hsv',)
new_pca = pca_dtset[:,:-3]
plt.show()


clusterNum = 3
k_means = KMeans(init='k-means++', n_clusters=clusterNum, n_init='auto')
k_means.fit(new_pca)
labels = k_means.labels_
print(labels)

wcss = []

for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++',n_init='auto')
    kmeans.fit(new_pca)
    wcss.append(kmeans.inertia_)
plt.plot (range(1, 15), wcss)
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()


plt.scatter(pca_dtset[:,0],pca_dtset[:,1],c=labels,cmap='hsv',label = labels)

plt.show()


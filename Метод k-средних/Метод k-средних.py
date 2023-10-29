import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Читаем файл,убираем текстовые данные
dt = pd.read_csv('data.csv')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 39)
dt = dt.drop(dt.columns[[1, 2, 3, 25, 27, 28, 31, 32]], axis=1)


# Добавляем нули в пропущенные строки
dt = dt.fillna(0)

# Проверка на отсутствующие данные
x = dt.isnull().sum()



x = dt.values[:, :]

print(dt)
x = np.nan_to_num(x)
dtset = StandardScaler().fit_transform(x)



clusterNum = 25
k_means = KMeans(init='k-means++', n_clusters=clusterNum, n_init=12)
k_means.fit(dtset)
labels = k_means.labels_
print(labels)

wcss = []
for i in range(1, 26):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(dtset)
    wcss.append(kmeans.inertia_)
plt.plot (range(1, 26), wcss)
plt.title('МеТОд Локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()


plt.scatter(x[:,22],x[:,26],c=labels,cmap='hsv',label = labels)
plt.xlim(1910,1990)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('glass.csv')
var_names = list(df.columns) #получение названий столбцов
labels = df.to_numpy('int')[:,-1] #последней цифры
data = df.to_numpy('float')[:,:-1] #информация о строках



# Предобработка
from sklearn import preprocessing
data = preprocessing.minmax_scale(data)

# print(data)

# графики
fig, axs = plt.subplots(2,4)
for i in range (data.shape[1]-1):
    axs[i // 4, i % 4].scatter(data[:,i],data[:,(i+1)], c = labels, cmap='hsv')
    axs[i//4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4]. set_ylabel(var_names[i+1])
    fig.set_size_inches(20, 10)
plt.show()
# график переделанный, от 1 и 2 компоненты
from sklearn.decomposition import PCA
pca = PCA(n_components =4)
pca_data = pca.fit(data).transform(data)
print(pca.explained_variance_ratio_)
# 4 комопоненты объяснят с шансом 85%
print(pca.singular_values_)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,cmap='hsv',)
plt.show()
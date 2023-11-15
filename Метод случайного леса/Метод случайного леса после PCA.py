import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

df = pd.read_csv('glass.csv')


var_names = list(df.columns) #получение названий столбцов
labels = df.to_numpy('int')[:,-1] #последней цифры
data = df.to_numpy('float')[:,:-1] #информация о строках
# print(data)
# Предобработка
data = preprocessing.minmax_scale(data)

# графики
fig, axs = plt.subplots(2,4)
for i in range (data.shape[1]-1):
    axs[i // 4, i % 4].scatter(data[:,i],data[:,(i+1)], c = labels, cmap='hsv')
    axs[i//4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4]. set_ylabel(var_names[i+1])
    fig.set_size_inches(20, 10)
plt.show()

# график переделанный, от 1 и 2 компоненты
pca = PCA(n_components =4)
pca_data = pca.fit(data).transform(data)

# 4 комопоненты объяснят с шансом 85%
print('4 компоненты объяснят с шансом: ',sum(pca.explained_variance_ratio_),' д.ед.')
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,cmap='hsv',)
plt.show()

# Распределяем новые данные на трейн и тест PCA
cdf = pca_data
new_cdf = np.insert(cdf,4, [labels],axis = 1)
print(new_cdf)
X = new_cdf[:,:-1]
y = new_cdf[:,-1]
train_X, test_X, train_y, test_y = train_test_split(X,y, test_size= 0.2)

sl = RandomForestRegressor(n_estimators=100, oob_score=True,random_state=1)
sl.fit (train_X,train_y)
a = sl.predict(test_X)
# print(a)

# Распределяем новые данные на трейн и тест без PCA
cdf1 = data
new_cdf1 = np.insert(cdf1,9, [labels],axis = 1)
X1 = new_cdf1[:,:-1]
y1 = new_cdf1[:,-1]
train_X1, test_X1, train_y1, test_y1 = train_test_split(X1,y1, test_size= 0.2)

sel = RandomForestRegressor(n_estimators=100, oob_score=False,random_state=1)
sel.fit (train_X1,train_y1)
a1 = sel.predict(test_X1)
# print(a1)


plt.subplot(1,2,1)
plt.scatter(test_y,a1, color="blue")
plt.title('Это первый график')
plt.subplot(1,2,2)
plt.title('Это второй график')
plt.scatter (test_y,a, color="green")
plt.show()

print('Accurancy on training set до PCA :',(sel.score(train_X1, train_y1)))
print('Accurancy on test_set до PCA :',(sel.score(test_X1,test_y1)))


print('Accurancy on training set после PCA :',(sl.score(train_X, train_y)))
print('Accurancy on test_set после PCA :',(sl.score(test_X,test_y)))


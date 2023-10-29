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
print(data)
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
print(pca.singular_values_)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,cmap='hsv',)
plt.show()

# Распределяем новые данные на трейн и тест PCA
cdf = pca_data
new_cdf = np.insert(cdf,4, [labels],axis = 1)
X = new_cdf[:,:-1]
y = new_cdf[:,-1]
train_X, test_X, train_y, test_y = train_test_split(X,y, test_size= 0.2)

sel = RandomForestRegressor(n_estimators=100, oob_score=True,random_state=1)
sel.fit (train_X,train_y)
a = sel.predict(test_X)
print(a)

# Распределяем новые данные на трейн и тест без PCA
cdf1 = data
new_cdf1 = np.insert(cdf1,9, [labels],axis = 1)
X1 = new_cdf1[:,:-1]
y1 = new_cdf1[:,-1]
train_X1, test_X1, train_y1, test_y1 = train_test_split(X1,y1, test_size= 0.2)

sel = RandomForestRegressor(n_estimators=100, oob_score=True,random_state=1)
sel.fit (train_X1,train_y1)
a1 = sel.predict(test_X1)
print(a1)
#
# plt.plot(1,a1, label = 'Без PCA')
# plt.plot(2,a, label = 'С PCA')
# plt.plot(3,test_y1, label = 'Тест для Без PCA')
# plt.plot(4,test_y, label = 'Тест для C PCA')
# plt.legend()
# plt.title ('Сверка данных')
# plt.show()

# print('AUC-ROC (oob) = ',roc_auc_score(y,sel.oob_prediction_, multi_class='ovr'))
# print('AUC-ROC (test) = ', roc_auc_score(test_y,a, multi_class='ovr'))
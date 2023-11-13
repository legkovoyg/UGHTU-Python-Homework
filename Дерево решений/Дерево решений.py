import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

df = pd.read_csv('glass.csv')
var_names = list(df.columns) #получение названий столбцов
labels = df.to_numpy('int')[:,-1] #последней цифры
data = df.to_numpy('float')[:,:-1] #информация о строках

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

# Распределяем новые данные на трейн и тест
cdf = pca_data
new_cdf = np.insert(cdf,4, [labels],axis = 1)
X = new_cdf[:,:-1]
y = new_cdf[:,-1]
train_X, test_X, train_y, test_y = train_test_split(X,y, test_size= 0.2)


# Обращаемся к дереву
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
get_n_leaves = 6
clf.fit(train_X, train_y)
y_pred =clf.predict(test_X)

# Проверка переобучения
print('Accurancy on training set :',format(clf.score(train_X, train_y)))
print('Accurancy on test_set:',format(clf.score(test_X,test_y)))


# смотрим на итоговое дерево
plt.subplots(1,1, figsize = (10,10))
tree.plot_tree(clf, filled = True)
plt.show()
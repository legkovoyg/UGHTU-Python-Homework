import numpy as np
import pandas as pd
# Беру датасет
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# делаю датасет
dataset = pd.read_csv(url, names=names)
# Смотрю на него
# print(dataset)
# Задаю х = первые 4 фактора, у = название цветка
x = dataset.iloc[: , 0:4].values
y = dataset.iloc[: , 4:5]
# Смотрю чему равно Х, чему У (проверка)
# print(X)
# print(y)
# Делим часть выборки на обучающую и тестовую
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts (x,y, test_size = 0.2, random_state = 0)
# Стандартизация данных
from sklearn.preprocessing import StandardScaler as SS
SC = SS()
x_train = SC.fit_transform(x_train)
x_test = SC.transform(x_test)
# Проверка
print (x_train)
# print(x_test)
# Переход к методу PCA


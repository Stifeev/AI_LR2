# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:23:33 2020

@author: stife
"""

#%% импорт всякого

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tree import tree

#%% Загрузка

PATH = os.path.join("datasets")
NAME = "winequality-red.csv"

def load_data(path = PATH, name = NAME): # вернуть DataFrame из csv
    csv_path = os.path.join(path, name)
    return pd.read_csv(csv_path)

wine = load_data()

#%% обзор данных в консоли

pd.options.display.max_columns = wine.shape[1]
print(wine.head())
print(wine.info())

print(wine.describe())

#%% обзор данных в гистограммах

wine.hist(bins = 8, figsize = (20, 20)) 
plt.show()

#%% проследим зависимости

corr_matrix = wine.corr()

print(corr_matrix["density"].sort_values(ascending = False))

#%% проследим зависимости

attributes = ["density", "fixed acidity", "citric acid", "residual sugar", "pH", "alcohol"] # наблюдаем хорошую линейную зависимость у fixed acidity и alcohol 
scatter_matrix(wine[attributes], figsize = (30, 30))

#%% попробуем понять как зависят "подозрительные аттрибуты"

print(corr_matrix["citric acid"]["fixed acidity"])

print(corr_matrix["total sulfur dioxide"]["free sulfur dioxide"])

#%% пробуем визуализировать данные - наблюдаем хорошую тенденденцию к разделению на страты

wine.plot(kind = "scatter", x = "fixed acidity", y = "alcohol", alpha = 0.6,
          s = wine["quality"] * 5, label = "quality", figsize = (10, 10),
          c = "density", cmap = plt.get_cmap ("jet"), colorbar = True)
plt.legend()

print(wine["quality"].value_counts())

wine["quality"], _ = wine["quality"].factorize(sort=True)

#%% отделим тестовую выборку от тренировочной с помощью стратифицированной выборки по quality

sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 42)
sss.get_n_splits(wine)

for train_i, test_i in sss.split(wine, wine["quality"]):
    train_set = wine.loc[train_i]
    test_set = wine.loc[test_i]

train_set.info()
test_set.info()

print(train_set["quality"].value_counts())
print(test_set["quality"].value_counts())

#%% в numpy-матрицы

print("Обучающих образцов:", train_set.shape[0])
print("Тестовых образцов:", test_set.shape[0])

y_train = np.int32(train_set["quality"])
y_test = np.int32(test_set["quality"])
X_train = np.float32(train_set[set(train_set.columns) - {"quality"}])
X_test = np.float32(test_set[set(test_set.columns) - {"quality"}])

#%% выполним масштабирование по минимаксу

scaler = MinMaxScaler((0, 1))
scaler.fit(np.concatenate((X_train, X_test), axis = 0))
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train[:5])

#%% построение модели KNN

def KNN_predict(X, y, target, k=5): # предсказание
    dist = []
    for i in range(target.shape[0]): 
        dist.append(np.sqrt(np.sum(np.square(X - target[i]), axis=1)))
    distanses = np.stack(dist) # (batch_size, train_samples)
    ans = np.zeros(target.shape[0], dtype=np.int32)
    for i in range(target.shape[0]):
        sort_k_neigbours = sorted(range(distanses.shape[1]), key=lambda x: distanses[i, x])[:k]
        counts = {i: 0 for i in set(y[sort_k_neigbours])}
        for j in sort_k_neigbours: # проходим по всем таким соседям
            counts[y[j]] += 1
        m = 0
        l = 0
        for j in counts:
            if counts[j] > m:
                m = counts[j]
                l = j
        ans[i] = l
    return ans

def accuracy(predict, y):
    return np.mean(np.where(predict == y, np.ones_like(y), np.zeros_like(y)))

#%% тестирование

for k in range(1, 10, 2):
    ans = KNN_predict(X_train, y_train, X_test, k=k)
    my_acc_val = accuracy(ans, y_test)
    ans = KNN_predict(X_train, y_train, X_train, k=k)
    my_acc_train = accuracy(ans, y_train)
    
    kn_clf = KNeighborsClassifier(n_neighbors=k)
    kn_clf.fit(X_train, y_train)
    
    ans = kn_clf.predict(X_test)
    sl_acc_val = accuracy(ans, y_test)
    ans = kn_clf.predict(X_train)
    sl_acc_train = accuracy(ans, y_train)
    
    print("Точность моей модели при k =", k, "на тренировочных данных:", my_acc_train)
    print("Точность моей модели при k =", k, "на тестовых данных:", my_acc_val)
    print("Точность модели sklearn при k =", k, "на тренировочных данных:", sl_acc_train)
    print("Точность модели sklearn при k =", k, "тестовых данных:", sl_acc_val)

#%% построение модели дерева решений

def J(l, r): # функция для максимизации
    return np.sum(l ** 2) / np.sum(l) + np.sum(r ** 2) / np.sum(r)

def DTC_train(X, y, min_samples_split=None, min_samples_leaf=None, max_depth=None): # обучение дерева
    """
    Препроцессинг перед рекурсией
    """
    n_classes = len(set(y))
    m = X.shape[0] # число образцов
    if min_samples_split is None:
        min_samples_split = int(np.round(m * 0.1))
    if min_samples_leaf is None:
        min_samples_leaf = int(np.round(m * 0.05))
    if max_depth is None:
        max_depth = n_classes + 1
    X = np.concatenate((X, y.reshape((-1, 1))), axis=1)
    root = tree() # корень дерева
    Split(root, X, 1, n_classes, min_samples_split, min_samples_leaf, max_depth)
    return root

def Split(root, X, height, n_classes, min_samples_split, min_samples_leaf, max_depth): # разбить корень на два
    m = X.shape[0] # число образцов
    n = X.shape[1] - 1 # число атрибутов
    """
    Этап подсчёта метрик
    """
    metrics = {}
    metrics["samples"] = m # число образцов
    values = np.zeros(n_classes, dtype=np.int32)
    for i in range(m):
        values[int(X[i, -1])] += 1
    metrics["values"] = values
    gini = 1 - np.sum((values / m) ** 2)
    metrics["gini"] = gini
    metrics["class"] = np.argmax(values)
    root.data = metrics # сохраняем метрики в узле
    if height >= max_depth or m < min_samples_split or m // 2 <= min_samples_leaf: # проверка ограничений
        return metrics
    X_sort = []
    for i in range(n):
        X_sort.append(sorted(X, key=lambda x: x[i])) # сортировка по i-му столбцу
    X = np.array(X_sort) # (n, m, n + 1) - тензор
    best_sol = {"f": -np.inf} # лучшее начальное решение для разбиения
    for i in range(n): # по всем атрибутам
        """
        Этап минимизации
        """
        l = np.zeros(n_classes, dtype=np.int32)
        r = values.copy()
        for j in range(m - min_samples_leaf): # по образцам
            l[int(X[i, j, -1])] += 1
            r[int(X[i, j, -1])] -= 1
            if min_samples_leaf <= j:
                cur_res = J(l, r)
                if cur_res > best_sol["f"]:
                    best_sol["f"] = cur_res
                    best_sol["k"] = i
                    best_sol["t"] = (X[i, j, i] + X[i, j + 1, i]) / 2
                    best_sol["i"] = j
    """
    Этап разбиения
    """
    i = best_sol["i"]
    k = best_sol["k"]
    XL = X[k][: i + 1]
    XR = X[k][i + 1: ]
    """
    Формирование результата и запуск рекурсии
    """
    root.key = (k, best_sol["t"])
    root.left = tree()
    root.right = tree()
    Split(root.left, XL, height + 1, n_classes, min_samples_split, min_samples_leaf, max_depth)
    Split(root.right, XR, height + 1, n_classes, min_samples_split, min_samples_leaf, max_depth)
    return metrics

def DTC_predict(root, X):
    y = np.zeros(X.shape[0], dtype=np.int32)
    for i in range(X.shape[0]):
        cur = root
        while not cur.is_leaf():
            k, t = cur.key
            if X[i, k] <= t:
                cur = cur.left
            else:
                cur = cur.right
        y[i] = cur.data["class"]
    return y

#%% re-скалирование
    
X_train = scaler.inverse_transform(X_train)
X_test = scaler.inverse_transform(X_test)

print(X_train[:5])

#%% Обучение моей модели
    
root = DTC_train(X_train, y_train, min_samples_leaf=1, max_depth=50, min_samples_split=2)
ans = DTC_predict(root, X_test)
accuracy = np.mean(np.where(ans == y_test, np.ones_like(ans), np.zeros_like(ans)))

print(accuracy)

#%% Обучение модели из библиотеки

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)

predict = tree_clf.predict(X_test)
sl_acc = np.mean(np.where(predict == y_test, np.ones_like(predict), np.zeros_like(predict)))

print("Точность моей модели:", accuracy)
print("Точность модели sklearn:", sl_acc)

#%% метрики

ans = DTC_predict(root, X_train)
acc_train = np.mean(np.where(ans == y_train, np.ones_like(ans), np.zeros_like(ans)))

predict = tree_clf.predict(X_test)
sl_acc_val = np.mean(np.where(predict == y_test, np.ones_like(predict), np.zeros_like(predict)))
predict = tree_clf.predict(X_train)
sl_acc_train = np.mean(np.where(predict == y_train, np.ones_like(predict), np.zeros_like(predict)))

print("Точность моей модели на тренировочных данных:", acc_train)
print("Точность моей модели на тестовых данных:", accuracy)
print("Точность модели sklearn на тренировочных данных:", sl_acc_train)
print("Точность модели sklearn тестовых данных:", sl_acc_val)
    

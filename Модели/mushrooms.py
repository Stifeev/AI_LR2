# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:38:28 2020

@author: stife
"""

#%% Импорт всякого

import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from time import sleep
from tqdm import tqdm
from scipy.optimize import LinearConstraint, minimize
from tree import tree

#%% Запустить в случае неполадок с прогресс баром

tqdm._instances.clear()

#%% Загрузка

PATH = "datasets"
NAME = "mushrooms.csv"

def load_data(path = PATH, name = NAME): # вернуть DataFrame из csv
    csv_path = os.path.join(path, name)
    return pd.read_csv(csv_path)

mushrooms = load_data()

#%% обзор данных в консоли

pd.options.display.max_columns = mushrooms.shape[1]
print(mushrooms.head())
mushrooms.info() # 23 категориальных признака 

#%% взглянем детальнее на каждый признак

print(mushrooms.describe())

for col in mushrooms: # 
    print(mushrooms[col].value_counts()) # обнаруживаем пропуски в stalk-root
    
# Нужно преобразовать каждый признак

#%% class {p, e} -> {0; 1}
    
print(mushrooms["class"].value_counts())
tr = {'e': 1, 'p': 0}
mushrooms["class"] = mushrooms["class"].map(tr)
print(mushrooms["class"][:10])

#%% cap-shape (каждое значение можно заменить числом, т.к. есть аналогия)

print(mushrooms["cap-shape"].value_counts())
tr = {'s': 0, 'f': 1, 'x': 2, 'k': 3, 'c': 3.3, 'b': 3.5}
mushrooms["cap-shape"] = mushrooms["cap-shape"].map(tr)
print(mushrooms["cap-shape"][:10])

#%% cap-surface

print(mushrooms["cap-surface"].value_counts())
tr = {'s': 0, 'f': 1, 'y': 2, 'g': 3}
mushrooms["cap-surface"] = mushrooms["cap-surface"].map(tr)
print(mushrooms["cap-surface"][:10])

#%% cap-color

print(mushrooms["cap-color"].value_counts())
tr_r = {'n': 165, 'b': 240, 'c': 210, 'g': 128, 'r': 0, 'p': 255, 'e': 255, 'w': 255, 'y': 255, 'u': 128}
tr_g = {'n': 42, 'b': 220, 'c': 105, 'g': 128, 'r': 128, 'p': 192, 'e': 0, 'w': 255, 'y': 255, 'u': 0}
tr_b = {'n': 42, 'b': 130, 'c': 40, 'g': 128, 'r': 0, 'p': 203, 'e': 0, 'w': 255, 'y': 0, 'u': 128}
mushrooms["cap-color-r"] = mushrooms["cap-color"].map(tr_r)
mushrooms["cap-color-g"] = mushrooms["cap-color"].map(tr_g)
mushrooms["cap-color-b"] = mushrooms["cap-color"].map(tr_b)
del mushrooms["cap-color"] # можно выкинуть
print(mushrooms["cap-color-r"][:10])
print(mushrooms["cap-color-g"][:10])
print(mushrooms["cap-color-b"][:10])
print(mushrooms.info())

#%% bruises

print(mushrooms["bruises"].value_counts())
tr = {'t': 1, 'f': 0}
mushrooms["bruises"] = mushrooms["bruises"].map(tr)
print(mushrooms["bruises"][:10])

#%% odor (запах). Этот признак нельзя соотнести с числом, поэтому закодируем его числом объектов у которых он есть

print(mushrooms["odor"].value_counts())
mushrooms["odor"] = mushrooms["odor"].map(mushrooms.groupby("odor").size())
print(mushrooms["odor"][:10])

#%% gill-attachment. Всё просто

print(mushrooms["gill-attachment"].value_counts())
tr = {'f': 0, 'a': 1}
mushrooms["gill-attachment"] = mushrooms["gill-attachment"].map(tr)
print(mushrooms["gill-attachment"][:10])

#%% gill-spacing. Всё просто

print(mushrooms["gill-spacing"].value_counts())
tr = {'c': 0, 'w': 1}
mushrooms["gill-spacing"] = mushrooms["gill-spacing"].map(tr)
print(mushrooms["gill-spacing"][:10])

#%% gill-color

print(mushrooms["gill-color"].value_counts())
tr_r.update({'k': 0, 'h': 123, 'o': 255})
tr_g.update({'k': 0, 'h': 63, 'o': 165})
tr_b.update({'k': 0, 'h': 0, 'o': 0})
mushrooms["gill-color-r"] = mushrooms["gill-color"].map(tr_r)
mushrooms["gill-color-g"] = mushrooms["gill-color"].map(tr_g)
mushrooms["gill-color-b"] = mushrooms["gill-color"].map(tr_b)
del mushrooms["gill-color"] # можно выкинуть
print(mushrooms["gill-color-r"][:10])
print(mushrooms["gill-color-r"][:10])
print(mushrooms["gill-color-r"][:10])

#%% gill-size

print(mushrooms["gill-size"].value_counts())
tr = {'n': 0, 'b': 1}
mushrooms["gill-size"] = mushrooms["gill-size"].map(tr)
print(mushrooms["gill-size"][:10])

#%% stalk-shape

print(mushrooms["stalk-shape"].value_counts())
tr = {'t': 0, 'e': 1}
mushrooms["stalk-shape"] = mushrooms["stalk-shape"].map(tr)
print(mushrooms["stalk-shape"][:10])

#%% stalk-root. Есть пропуски - применим one-hot-encoder

print(mushrooms["stalk-root"].value_counts())
s = set(mushrooms["stalk-root"].unique())
s.remove('?')
for i in s:
    mushrooms["stalk-root" + '=' + i] = np.float32(mushrooms["stalk-root"] == i)
    print(mushrooms["stalk-root" + '=' + i][:10])
del mushrooms["stalk-root"]

#%% stalk-surface-above-ring

print(mushrooms["stalk-surface-above-ring"].value_counts())
tr = {'s': 0, 'k': 1, 'f': 2, 'y': 3}
mushrooms["stalk-surface-above-ring"] = mushrooms["stalk-surface-above-ring"].map(tr)
print(mushrooms["stalk-surface-above-ring"][:10])

#%% stalk-surface-below-ring

print(mushrooms["stalk-surface-below-ring"].value_counts())
tr = {'s': 0, 'k': 1, 'f': 2, 'y': 3}
mushrooms["stalk-surface-below-ring"] = mushrooms["stalk-surface-below-ring"].map(tr)
print(mushrooms["stalk-surface-below-ring"][:10])

#%% stalk-color-above-ring

print(mushrooms["stalk-color-above-ring"].value_counts())
mushrooms["stalk-color-above-ring-r"] = mushrooms["stalk-color-above-ring"].map(tr_r)
mushrooms["stalk-color-above-ring-g"] = mushrooms["stalk-color-above-ring"].map(tr_g)
mushrooms["stalk-color-above-ring-b"] = mushrooms["stalk-color-above-ring"].map(tr_b)
del mushrooms["stalk-color-above-ring"] # можно выкинуть
print(mushrooms["stalk-color-above-ring-r"][:10])
print(mushrooms["stalk-color-above-ring-g"][:10])
print(mushrooms["stalk-color-above-ring-b"][:10])

#%% stalk-color-below-ring

print(mushrooms["stalk-color-below-ring"].value_counts())
mushrooms["stalk-color-below-ring-r"] = mushrooms["stalk-color-below-ring"].map(tr_r)
mushrooms["stalk-color-below-ring-g"] = mushrooms["stalk-color-below-ring"].map(tr_g)
mushrooms["stalk-color-below-ring-b"] = mushrooms["stalk-color-below-ring"].map(tr_b)
del mushrooms["stalk-color-below-ring"] # можно выкинуть
print(mushrooms["stalk-color-below-ring-r"][:10])
print(mushrooms["stalk-color-below-ring-g"][:10])
print(mushrooms["stalk-color-below-ring-b"][:10])

#%% veil-type. Бесполезный признак

print(mushrooms["veil-type"].value_counts())
del mushrooms["veil-type"]

#%% veil-color

print(mushrooms["veil-color"].value_counts())
mushrooms["veil-color-r"] = mushrooms["veil-color"].map(tr_r)
mushrooms["veil-color-g"] = mushrooms["veil-color"].map(tr_g)
mushrooms["veil-color-b"] = mushrooms["veil-color"].map(tr_b)
del mushrooms["veil-color"] # можно выкинуть
print(mushrooms["veil-color-r"][:10])
print(mushrooms["veil-color-g"][:10])
print(mushrooms["veil-color-b"][:10])

#%% ring-number

print(mushrooms["ring-number"].value_counts())
tr = {'n': 0, 'o': 1, 't': 2}
mushrooms["ring-number"] = mushrooms["ring-number"].map(tr)
print(mushrooms["ring-number"][:10])

#%% ring-type. Закодируем числом экземпляров

print(mushrooms["ring-type"].value_counts())
mushrooms["ring-type"] = mushrooms["ring-type"].map(mushrooms.groupby("ring-type").size())
print(mushrooms["ring-type"][:10])

#%% spore-print-color

print(mushrooms["spore-print-color"].value_counts())
mushrooms["spore-print-color-r"] = mushrooms["spore-print-color"].map(tr_r)
mushrooms["spore-print-color-g"] = mushrooms["spore-print-color"].map(tr_g)
mushrooms["spore-print-color-b"] = mushrooms["spore-print-color"].map(tr_b)
del mushrooms["spore-print-color"] # можно выкинуть
print(mushrooms["spore-print-color-r"][:10])
print(mushrooms["spore-print-color-r"][:10])
print(mushrooms["spore-print-color-r"][:10])

#%% population

print(mushrooms["population"].value_counts())
tr = {'y': 1, 's': 2, 'v': 3, 'c': 4, 'a': 5, 'n': 6}
mushrooms["population"] = mushrooms["population"].map(tr)
print(mushrooms["population"][:10])

#%% habitat (one-hot-encoder)

print(mushrooms["habitat"].value_counts())

for i in mushrooms["habitat"].unique():
    mushrooms["habitat" + '=' + i] = np.float32(mushrooms["habitat"] == i)
    print(mushrooms["habitat" + '=' + i][:10])
    
del mushrooms["habitat"]

#%% результат проделанной работы

mushrooms.info()

#%% отделим тестовую выборку от тренировочной с помощью стратифицированной выборки по ring-type

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 42)
sss.get_n_splits(mushrooms)

for train_i, test_i in sss.split(mushrooms, mushrooms["ring-type"]):
    train_set = mushrooms.loc[train_i]
    test_set = mushrooms.loc[test_i]

train_set.info()
test_set.info()

print(train_set["ring-type"].value_counts())
print(test_set["ring-type"].value_counts())

#%% в numpy-матрицы

print("Обучающих образцов:", train_set.shape[0])
print("Тестовых образцов:", test_set.shape[0])

y_train = np.int32(train_set["class"])
y_test = np.int32(test_set["class"])
X_train = np.float32(train_set[set(train_set.columns) - {"class"}])
X_test = np.float32(test_set[set(test_set.columns) - {"class"}])

#%% выполним масштабирование

scaler = StandardScaler()
scaler.fit(np.concatenate([X_train, X_test], axis=0)) # я решил сделать fit на всём датасете
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train[:5])

#%% в мини-пакеты

batch_size = 50 # размер мини-пакета

batches = []

for i in range(0, X_train.shape[0], batch_size):
    batches.append((X_train[i: i + batch_size], y_train[i: i + batch_size]))
    
n_batches = len(batches)

print("Всего", n_batches, "обучающих минипакетов по", batch_size, "образцов")

#%% Граф вычислений для ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ

tf.reset_default_graph()

n_inputs = batches[0][0].shape[1]

X = tf.placeholder(tf.float32, [None, n_inputs]) # образец (batch_size, n_inputs)
y = tf.placeholder(tf.float32, [None]) # цель (batch_size)

stddev = tf.constant(2 / np.sqrt(n_inputs + 1), dtype=tf.float32) # стандартное отклонение
tetta = tf.Variable(tf.truncated_normal([n_inputs], stddev=stddev, dtype=tf.float32, seed=42)) # инициализация весов с помощью усечённого гауссова распределения (n_inputs)
b = tf.Variable(0, dtype=tf.float32) # член смещения

sigma = lambda t: 1 / (1 + tf.exp(-t)) # логистическая функция

p = sigma(tf.tensordot(X, tetta, 1) + b) # оценка вероятности (batch_size)

loss = - tf.reduce_mean(y * tf.log(p) + (1 - y) * tf.log(1 - p)) # функция потерь

learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate)

training_op = optimizer.minimize(loss)

predict = tf.where(tf.less(p, 0.5), tf.zeros_like(p), tf.ones_like(p)) # предсказание модели

accuracy = tf.reduce_mean(tf.where(tf.equal(predict, y), tf.ones_like(p), tf.zeros_like(p))) # точность

init = tf.global_variables_initializer()
saver = tf.train.Saver()

save_path = os.path.join("LR_save", "weigths.ckpt")

#%% Обучение моей модели

epoches = 4

with tf.Session() as sess:
    init.run()
    for i in range(epoches):
        n_batches = len(batches)
        indicies = np.random.permutation(range(n_batches))
        tq = tqdm(range(n_batches), desc = "Эпоха " + str(i + 1))
        acc_val_avg = 0
        acc_train_avg = 0
        for j in tq:
            X_batch, y_batch = batches[indicies[j]]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test}) # точность на тестовом сете
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch}) # точность на текущем мини-пакете
            acc_val_avg += acc_val
            acc_train_avg += acc_train
            if j < n_batches - 1:
                tq.set_postfix_str("val acc = " + str(acc_val) + " train acc = " + str(acc_train)) 
            else:
                tq.set_postfix_str("avg val acc = " + str(np.round(acc_val_avg / n_batches, 5)) + 
                                   " avg train acc = " + str(np.round(acc_train_avg / n_batches, 5)))
            #sleep(0.05) # УБРАТЬ, ЕСЛИ НЕ ХОТИТЕ ДЕТАЛЬНЫЙ ПРОГРЕСС
        tq.close()
    save_path = saver.save(sess, save_path) # сохранение весов 
        
#%% Обучение модели из sklearn
        
log_rec = LogisticRegression()
log_rec.fit(X_train, y_train)

#%% Вычисление метрик

my_acc_val = 0 # точность моей модели
my_acc_train = 0
sl_acc_val = 0 # точность модели из библиотеки
sl_acc_train = 0 

with tf.Session() as sess:
    saver.restore(sess, save_path)
    my_acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
    my_acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
    
predict = log_rec.predict(X_test)
sl_acc_val = np.mean(np.where(predict == y_test, np.ones_like(predict), np.zeros_like(predict)))
predict = log_rec.predict(X_train)
sl_acc_train = np.mean(np.where(predict == y_train, np.ones_like(predict), np.zeros_like(predict)))

print("Точность моей модели на тренировочных данных:", my_acc_train)
print("Точность моей модели на тестовых данных:", my_acc_val)
print("Точность модели sklearn на тренировочных данных:", sl_acc_train)
print("Точность модели sklearn тестовых данных:", sl_acc_val)
        
#%% Модель SVM с жёстким зазором
        
n_inputs = X_train.shape[1] # число входных признаков

np.random.seed(42)
stddev = 2 / np.sqrt(n_inputs + 1) # стандартное отклонение
W = np.random.normal(size = n_inputs, scale=stddev) # инициализация весов с помощью усечённого гауссова распределения (n_inputs)
b = 0

def SVM_predict(X): # прогноз линейного классификатора SVM
    return np.where(np.dot(X, W + b) < 0, np.zeros(X.shape[0]), np.ones(X.shape[0]))

def f(x): # функция для оптимизации
    return np.sum((x[:-1] ** 2)) / 2

def grad(x): # её градиент
    return np.concatenate((x[:-1], [0]))

def hess(x): # её матрица Гесса
    return np.diag(np.concatenate((np.ones(len(x) - 1), [0])))

def SVM_train(X, y, maxiter=10): # обучение
    global W, b
    m = X.shape[0] # число образцов
    t = np.where(y == 0, -np.ones_like(y), np.ones_like(y)).reshape((-1, 1))
    A = np.hstack((X, b * np.ones([m, 1])))
    A = t * A
    linear_constraint = LinearConstraint(A, np.ones(m), np.ones(m) * np.inf) # ограничения на переменные
    x0 = np.concatenate((W, [b]))
    res = minimize(f, x0, method='trust-constr', jac=grad, hess=hess,
                constraints=[linear_constraint],
                options={'verbose': 2, 'maxiter': maxiter})
    W = res.x[:-1]
    b = res.x[-1]
     
#%% Обучение моей модели (может занять много времени, из-за задачи УСЛОВНОЙ минимизации пакетное обучение не работает)

SVM_train(X=X_train, y=y_train)
acc_val = np.mean(np.where(SVM_predict(X_test) == y_test, 
                   np.ones_like(y_test), 
                   np.zeros_like(y_test)))
acc_train = np.mean(np.where(SVM_predict(X_train) == y_train, 
                   np.ones_like(y_train), 
                   np.zeros_like(y_train)))
print("Точность на тренировочных данных:", acc_train)
print("Точность на тестовых данных:", acc_val)

#%% Обучение модели из sklearn с теми же параметрами
        
svm_clf = LinearSVC(C=1, loss="hinge")
svm_clf.fit(X_train, y_train)

#%% Вычисление метрик

predict = svm_clf.predict(X_test)
sl_acc_val = np.mean(np.where(predict == y_test, np.ones_like(predict), np.zeros_like(predict)))
predict = svm_clf.predict(X_train)
sl_acc_train = np.mean(np.where(predict == y_train, np.ones_like(predict), np.zeros_like(predict)))

print("Точность моей модели на тренировочных данных:", acc_train)
print("Точность моей модели на тестовых данных:", acc_val)
print("Точность модели sklearn на тренировочных данных:", sl_acc_train)
print("Точность модели sklearn тестовых данных:", sl_acc_val)

#%% Моё дерево решений

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
    
root = DTC_train(X_train, y_train, min_samples_leaf=10, max_depth=30, min_samples_split=5)
ans = DTC_predict(root, X_test)
accuracy = np.mean(np.where(ans == y_test, np.ones_like(ans), np.zeros_like(ans)))

print(accuracy)

#%% Модель из библиотеки

tree_clf = DecisionTreeClassifier(min_samples_leaf=10, max_depth=30, min_samples_split=5)
tree_clf.fit(X_train, y_train)

#%% Метрики

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


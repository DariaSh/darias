
# coding: utf-8

# Имеется некоторая выборка из нормального распределения размера n и известными средним и дисперсией = 5. Предположим мы проверяем простую двухстороннюю гипотезу о равенстве среднего μ = 0 (μ!= 0 - альтернатива)), т.е. H0:μ=0 and H1:μ≠0  с заданным вещественным параметром alpha = 0.05, задающим область принятия гипотезы. p <= alpha - H0 отклоняется, p > alpha, H0 - недостаточно оснований отклонить H0.
# Ошибки первого рода: отклоняем верную нулевую гипотезу
# Ошибки второго рода:  не отклоняем H0 когда она неверна
# 
# 1. Для того чтобы имитировать ошибки первого рода сгенерируем распределение, соответствующие нулевой гипотезе H0
# 2. Для того чтобы имитировать ошибки второго рода сгенерируем распределение, соответствующие альтернативной гипотезе H1.

import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt


N = 10000
n = 100
alpha = 0.05
s = 5.
m0 = 0.
m1 = 2.5


t1_errors = []
err1_rate = []
c1 = 0
for i in range(N):
    x = np.array([np.random.normal(m0, s) for i in range(n)])
    t, p = st.ttest_1samp(x, 0.)
    if p <= alpha:
        c1 +=1
    t1_errors.append(c1)
    err1_rate.append((c1/len(t1_errors)*100))
        
t2_errors = []
err2_rate = []
c2 = 0
for i in range(N):
    x = np.array([np.random.normal(m1, s) for i in range(n)])
    t, p = st.ttest_1samp(x, 0.)
    if p > alpha:
        c2+=1
    t2_errors.append(c2)
    err2_rate.append((c2/len(t2_errors)*100))

x = range(-n, n)

pdf1 = st.norm.pdf(x, m0, s)
plt.plot(x, pdf1, c = 'r')
pdf2 = st.norm.pdf(x, m1, s)
plt.plot(x, pdf2, c = 'b')
plt.title("Распределения для проверки гипотез")
plt.show()

t1 = np.array(t1_errors)
t2 = np.array(t2_errors)
er1 = np.array(err1_rate)
er2 = np.array(err2_rate)

plt.plot(t1, c= 'b')
plt.title("Ошибки первого рода")
plt.show()
plt.plot(t2, c= 'r')
plt.title("Ошибки второго рода")
plt.show()
plt.plot(er1, c= 'b')
plt.title("Доля ошибок первого рода")
plt.show()
plt.plot(er2, c= 'r')
plt.title("Доля ошибок второго рода")
plt.show()

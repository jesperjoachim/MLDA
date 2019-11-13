# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
def stripChar(string, char_to_remove=[]):
    print(char_to_remove)
    for char in char_to_remove:
        print(char)
        if char in string:
            print(char, string)
            string.replace(char, "")
    return string


hey = stripChar("100M", char_to_remove=["M"])
print(hey)

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.feature_selection import f_regression, mutual_info_regression 
from scipy import stats

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# -

# Input parameters: x1, x2, x3

# +
np.random.seed(0)
X = np.random.rand(1000, 3)

# print(X[:1, :])
x1 = X[:, 0]
x2 = X[:, 1]
x3 = X[:, 2]
serie1 = pd.Series(x2)
serie1.values
# print(len(X))
# print(len(x1))
# print(X)
# print(x1)
# -

# Output variable or dependent variable, y. Based on  the functions fx12/fx123.
# The fx123 includes the stochastic generated noise

# +
def fx12(x1, x2):
    return 1*x1 + np.sin(6*np.pi*x2)

def fx123(x1, x2, x3):
    return 1*x1 + np.sin(6*np.pi*x2) + .1*np.random.randn(1000)


# -

# Calc of Stats1 - i.e. __F-test__ (linear-like)/ __MI__ (nonlinear)
#
# Excerpts from wiki: 
# 1) __F-test__: It is most often used when comparing statistical models that have been fitted to a data set, in order to identify the model that best fits the population from which the data were sampled
# 2) __MI__: the mutual information (MI) of two random variables is a measure of the mutual dependence between the two variable

len(x1)
# Reshape if passing in single dimension, 1D
mi_ = mutual_info_regression(x2.reshape(-1,1), fx123(x1,x2,x3))
# print([x1])
# print((fx123(x1,x2,x3)))
mi_ 
# mi /= np.max(mi)
# f_test

# +
X
# fx123(x1, x2, x3)
f_test, _ = f_regression(X, fx123(x1, x2, x3))
f_test /= np.max(f_test)

mi = mutual_info_regression(X, fx123(x1,x2,x3))
mi
mi /= np.max(mi)
# f_test
# -

# Calc of Stats2 - i.e. __Pearson's correlation__ (linear relationships)/ __Spearman rank correlation__ n(nonparametric). Non-parametric means exluding the mean and variance, for example.
#
# Excerpts from wiki: 
# 1) __Pearson's correlation__: Pearson's correlation is a measure of the linear correlation between two variables X and Y.
# 2) __Spearman rank correlation__: A Spearman correlation of 1 results when the two variables being compared are monotonically related, even if their relationship is not linear. This means that all data-points with greater x-values than that of a given data-point will have greater y-values as well. In contrast, this does not give a perfect Pearson correlation.

# +
stats.pearsonr(X[:,0], fx123(x1, x2, x3))
for i in range(len(X[0])):
    print(i)

pearson = [stats.pearsonr(X[:,i], fx123(x1, x2, x3)) for i in range(len(X[0]))]
# pearson

spearman = [stats.spearmanr(X[:,i], fx123(x1, x2, x3)) for i in range(len(X[0]))]
# spearman
# -

plt.figure(figsize=(15,5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.scatter(X[:, i], fx123(x1,x2,x3), edgecolor='black', s=20)
    plt.xlabel("$x{}$".format(i + 1), fontsize=14)
    if i == 0:
        plt.ylabel('$y$', fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]), 
             fontsize=15)
plt.show();

np.random.random(1)[0]

# ### Creating a (perfect) surface plot and plotting the scattered values upon that###
#
# From a predefined x1,x2-mesh we calc the resulting y-values from: 1*x1 + np.sin(6*np.pi*x2) fx12(x1,x2)
# This result in a perfect xyz-surface plot.
# On this plot we plot our (actual) values calc from fx123(x1,x2, x3)

# +
plt.figure(figsize=(15,10))

plt.subplot(1, 2, 1)
x1_mesh = np.linspace(0, 1, 1000)
x2_mesh = np.linspace(0, 1, 1000)

X, Y = np.meshgrid(x1_mesh, x2_mesh)

# ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)

Z = fx12(X, Y)

ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='viridis')
ax.set_xlabel('x1')

ax.set_ylabel('x2')
ax.set_zlabel('z');

# ax = plt.axes(projection='3d')
ax.scatter(x1, x2, fx123(x1,x2,x3), c=fx123(x1,x2,x3), cmap='viridis', linewidth=0.5);

ax.view_init(15, 170)
plt.show();

# -

# ### Surface Triangulations:###
#
# As an alternative to the above plot we can use surface triangulation
#
# Excerpt from __Python Data Science Handbook by Jake VanderPlas__: For some applications, the evenly sampled grids required by the preceding routines are overly restrictive and inconvenient. In these situations, the triangulation-based plots can be very useful. What if rather than an even draw from a Cartesian or a polar grid, we instead have a set of random draws?

# +
ax = plt.axes(projection='3d')

ax.scatter(x1, x2, fx123(x1, x2, x3), c=fx123(x1, x2, x3), cmap='viridis', linewidth=0.5);
# -

# This leaves a lot to be desired. The function that will help us in this case is ax.plot_trisurf, which creates a surface by first finding a set of triangles formed between adjacent points (the result is shown in Figure 4-100; remember that x, y, and z here are one-dimensional arrays):

# +
plt.figure(figsize=(15,10))

ax = plt.axes(projection='3d')
ax.plot_trisurf(x1, x2, fx123(x1, x2, x3), cmap='viridis', edgecolor='none')

ax.view_init(15, 170)
plt.show();
# -

# ### Removing artefacts###
# Depending on the dataset and the algorithm used for the triangulation, some triangles might cause artifact in the 3D surface. In case that happens, it is possible to mask out some of them. In this example we eliminate the triangles for which at least one point is outside of an arbitrary boundary

# +
# isBad is an Array[points,] which contains for each (x,y) coordinate a boolean value indicating whether the point is outside (True) or inside (False) of a boundary condition.
# mask is an Array[nrOfTriangles,] in which each boolean value indicates whether the respective triangles was defined using at least one point outside of the boundary.

isBad = np.where((x1<0.001) | (x1>0.99) | (x2<0.001) | (x2>0.99), True, False)

mask = np.any(isBad[triang.triangles],axis=1)
triang.set_mask(mask)

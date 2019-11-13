import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# import seaborn as sns
# import statsmodels.api as sm
# import logging.config


# from mpl_toolkits.mplot3d import axes3d
# from sklearn import preprocessing
# from sklearn.feature_selection import f_regression, mutual_info_regression
# from scipy import stats
# from statsmodels.formula.api import ols

from MLDA.corr_stats import stat_functions as sf

from MLDA.corr_stats import stat_helperfunc as shf


def zero_replace(array):
    """Approved
    Part 0 of 3 - Helper function"""
    for r in range(array.shape[0]):
        for c in range(array.shape[1]):
            if array[r][c] == 0:
                array[r][c] = 1
    return array


def removeNan(serie1, serie2):
    df = pd.concat([serie1, serie2], axis=1)
    df_drop = df.dropna()
    return df_drop.iloc[:, 0], df_drop.iloc[:, 1]


def checkAllRowsCatOrNum(serie, choose_type="cat"):
    """Description..
    Input: Pandas series and type to check, string (cat or num). 
    Output: ..."""
    pass


def corrType(type1, type2):
    if type1 == "cat" and type2 == "cat":
        return "catcat"
    elif type1 == "num" and type2 == "num":
        return "numnum"
    else:
        return "catnum"


def corrMethodsExecute(corr_type, method_cc, method_cn, method_nn):
    dict_methods = {"catcat": method_cc, "catnum": method_cn, "numnum": method_nn}
    return dict_methods[corr_type]


def corrValue(serie1, serie2, methods):
    """Returns the asym values of serie1/2 (i.e 2 diff values), or the max values (same)
    for the symmetric case"""
    corr_values = []
    if "Asym" in methods:
        # asymmetric matrix
        method = methods[0]
        corr_values.extend(
            (
                sf.function_dict[method](serie1, serie2),
                sf.function_dict[method](serie2, serie1),
            )
        )
    elif methods:
        # symmetric matrix
        value = []
        # loop through the stated methods to be evaluated
        for method in methods:
            value.append(sf.function_dict[method](serie1, serie2)[0])
        # take the max corr_value from the evaluated methods
        value_max = max(value)
        corr_values.extend((value_max, value_max))
    else:
        return "error in methods"
    return corr_values


def findCorr(
    serie1,
    serie2,
    catcol,
    method_cc=["Asym"],
    method_cn=["Omega"],
    method_nn=["Spear", "Pear"],
):
    if serie1.name == serie2.name:
        corr_value = (1, 1)
    else:
        type_serie1 = ["cat" if serie1.name in catcol else "num"][0]
        type_serie2 = ["cat" if serie2.name in catcol else "num"][0]
        corr_type = corrType(type_serie1, type_serie2)
        corr_methods = corrMethodsExecute(corr_type, method_cc, method_cn, method_nn)
        corr_value = corrValue(serie1, serie2, corr_methods)
    return corr_value


def correlation(
    dataset,
    catcols,
    numcols,
    method_cc=["Asym"],
    method_cn=["Omega"],
    method_nn=["Spear", "Pear"],
):
    """Returns association strength matrix of any combination between numerical and categorical variables"""
    # Rearrange dataset so categoric columns are to the right
    dataset = dataset[catcols + numcols]
    columns = dataset.columns
    # constructing a square matrix and looping through each row and columns
    corr_matrix = pd.DataFrame(index=columns, columns=columns)
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            serie1, serie2 = removeNan(dataset[columns[i]], dataset[columns[j]])
            corr_values = findCorr(
                serie1,
                serie2,
                catcols,
                method_cc=method_cc,
                method_cn=method_cn,
                method_nn=method_nn,
            )
            # Add calc values to each cell pairs
            corr_matrix.iloc[i][j] = corr_values[0]
            corr_matrix.iloc[j][i] = corr_values[1]
    return corr_matrix


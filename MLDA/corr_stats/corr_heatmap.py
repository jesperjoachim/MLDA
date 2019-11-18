import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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


def checkdtypeOfColumns(catcols, numcols, df):
    """Function to check that columns are of the correct dtype - categorical data/columns should be 'category', numerical should be float64 or int64.
    Input: categorical and numerical columns as names (syntax: 'df[entry]') or integers (syntax: 'df.iloc[:, entry]'), df: pandas dataframe
    Output: 'ok' or 'which FIRST column is wrong'"""
    if str((catcols + numcols)[0]).isnumeric():
        # pandas syntax when catcols/numcols is entered as integers
        out = f"dtype of categorical columns: {catcols} and numerical columns {numcols} are ok"
        for entry in catcols:
            if df.iloc[:, entry].dtype.name != "category":
                out = f"The column '{entry}' is NOT category"
                return out
        for entry in numcols:
            if not (
                df.iloc[:, entry].dtype.name == "float64"
                or df.iloc[:, entry].dtype.name == "int64"
            ):
                out = f"The column '{entry}' is NOT float64/int64"
                return out
        return out
    else:
        # pandas syntax when catcols/numcols is entered as column names
        out = f"dtype of categorical columns: {catcols} and numerical columns {numcols} are ok"
        for entry in catcols:
            if df[entry].dtype.name != "category":
                out = f"The column '{entry}' is NOT category"
                return out
        for entry in numcols:
            if not (
                df[entry].dtype.name == "float64" or df[entry].dtype.name == "int64"
            ):
                out = f"The column '{entry}' is NOT float64/int64"
                return out
        return out


def removeNan(serie1, serie2):
    df = pd.concat([serie1, serie2], axis=1)
    df_drop = df.dropna()
    return df_drop.iloc[:, 0], df_drop.iloc[:, 1]


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


# def corrValue(serie1, serie2, methods):
#     """Returns the asym values of serie1/2 (i.e 2 diff values), or the max values (same)
#     for the symmetric case"""
#     corr_values = []
#     if "Asym" in methods:
#         # asymmetric matrix
#         method = methods[0]
#         corr_values.extend(
#             (
#                 sf.function_dict[method][0](serie1, serie2),
#                 sf.function_dict[method][0](serie2, serie1),
#             )
#         )
#     elif methods:
#         # symmetric matrix
#         value = []
#         # loop through the stated methods to be evaluated
#         for method in methods:
#             value.append(sf.function_dict[method][0](serie1, serie2)[0])
#         # take the max corr_value from the evaluated methods
#         value_max = max(value)
#         corr_values.extend((value_max, value_max))
#     else:
#         return "error in methods"
#     return corr_values


def screenCorrValuesBasedOnPvalue(corr_value, p_value, CI=0.1):
    if p_value < CI:
        return corr_value
    else:
        return "p > CI"


def findCorrPvalBasedAndNot(serie1, serie2, method):
    """Splitting function for getting correlation and p-values. Based on input it chooses either: 
    1) p-value based methods or 2) non-p-value based methods
    Input: Two pandas series, and correlation method
    Output: 2 x two-value tuple, for example if asym: ((corr12, p12), (coor21, p21)). For sym corr12=corr21 and p12=p21"""

    if sf.function_dict[method][1] == "no_pvalue":
        corr_and_pvalues = sf.calcCorrAndMimicP(serie1, serie2, method)
    elif sf.function_dict[method][1] == "with_pvalue":
        corr_and_pvalues = sf.calcCorrAndP(serie1, serie2, method)
        corr_and_pvalues = corr_and_pvalues, corr_and_pvalues
    return corr_and_pvalues


def findCorrSelectMethod(
    serie1, serie2, catcol, method_cc="Asym", method_cn="Omega", method_nn="Pear"
):
    if serie1.name == serie2.name:
        corr_and_pvalues = ((1, 1), (1, 1))
    else:
        type_serie1 = ["cat" if serie1.name in catcol else "num"][0]
        type_serie2 = ["cat" if serie2.name in catcol else "num"][0]
        corr_type = corrType(type_serie1, type_serie2)
        corr_method = corrMethodsExecute(corr_type, method_cc, method_cn, method_nn)
        corr_and_pvalues = findCorrPvalBasedAndNot(serie1, serie2, corr_method)

    # Lastly, we screen corr_values based on pvalues and restore them in the variable corr_and_pvalues
    # corr_values12 = screenCorrValuesBasedOnPvalue(
    #     corr_and_pvalues[0][0], corr_and_pvalues[0][1]
    # )
    # corr_values21 = screenCorrValuesBasedOnPvalue(
    #     corr_and_pvalues[1][0], corr_and_pvalues[1][1]
    # )
    # corr_and_pvalues = (
    #     corr_values12,
    #     corr_and_pvalues[0][1],
    #     (corr_values21, corr_and_pvalues[1][1]),
    # )
    return corr_and_pvalues


def correlation(
    dataset, catcols, numcols, method_cc="Cramer_V", method_cn="Omega", method_nn="Pear"
):
    """Returns association strength matrix of any combination between numerical and categorical variables
    Matrices with both corelation and p-values"""
    # Arrange dataset so categoric columns are to the right

    dataset = dataset[catcols + numcols]
    columns = dataset.columns
    # constructing a square matrix and looping through each row and columns
    asso_matrix_corr = pd.DataFrame(index=columns, columns=columns)
    asso_matrix_p = pd.DataFrame(index=columns, columns=columns)
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            serie1, serie2 = removeNan(dataset[columns[i]], dataset[columns[j]])
            corr_and_pvalues = findCorrSelectMethod(
                serie1,
                serie2,
                catcols,
                method_cc=method_cc,
                method_cn=method_cn,
                method_nn=method_nn,
            )
            # Add calc values to each cell pairs in the matrices
            asso_matrix_corr.iloc[i][j] = corr_and_pvalues[0][0]
            asso_matrix_corr.iloc[j][i] = corr_and_pvalues[1][0]
            asso_matrix_p.iloc[i][j] = corr_and_pvalues[0][1]
            asso_matrix_p.iloc[j][i] = corr_and_pvalues[1][1]
    return asso_matrix_corr, asso_matrix_p


# df_test = pd.read_excel("/home/jesper/Work/macledan/input_files/test_DF.xlsx")

# serie1, serie2 = df_test["X1"], df_test["Y1"]
# actual = calcCorrAndP(serie1, serie2, "Spear")
# print(actual)

# #
# ,
# catcols = ["global_warm_risk", "gender", "intimacy", "group", "num_police"]
# numcols = ["weight", "X3", "Y1"]
# for entry in catcols:
#     df_test[entry] = df_test[entry].astype("category")

# print(checkdtypeOfColumns(catcols, numcols, df_test))
# print(correlation(df_test, catcols, numcols, method_cc="Cramer_V"))


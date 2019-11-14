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


def checkdtypeOfColumns(catcols, numcols, df):
    """Description..
    Input: .
    Output: ..."""
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


# testing value 1 of a list with 2 values
tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
serie1, serie2 = tab1_input["num_police"], tab1_input["intimacy"]
result = corrValue(serie1, serie2, ["Asym"])
print(result)
# self.assertAlmostEqual(0.019759842721107736, result[0], places=4)


# df_test = pd.read_excel("/home/jesper/Work/macledan/input_files/test_DF.xlsx")
# print(df_test)

# catcols = ["global_warm_risk", "intimacy", "gender", "group", "num_police"]
# numcols = ["X1", "weight", "X3", "Y1"]
# for entry in catcols:
#     df_test[entry] = df_test[entry].astype("category")
# df_test.info()
# # print((df_test["gender"].dtype.name) == "category")

# print(checkdtypeOfColumns(catcols, numcols, df_test))

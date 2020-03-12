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


def screenCorrValuesBasedOnPvalue(corr_and_p_value, CI=0.1):
    """Function for screening corr-values based on p-values. If p-val > CI replace corr-val with 'p>CI',
    else keep corr-val.
    Input: double pairs of corr-val and p-val - i.e. 2 x two-value tuple.
    Output: same as input."""
    # First pair in tuple
    if corr_and_p_value[0][1] < CI:
        corr12 = corr_and_p_value[0][0]
    elif corr_and_p_value[0][1] > CI:
        corr12 = "p > CI"
    else:
        corr12 = np.NaN
    # Second pair in tuple
    if corr_and_p_value[1][1] < CI:
        corr21 = corr_and_p_value[1][0]
    elif corr_and_p_value[1][1] > CI:
        corr21 = "p > CI"
    else:
        corr21 = np.NaN
    # Recombine screened values
    corr_and_p_value = (
        (corr12, corr_and_p_value[0][1]),
        (corr21, corr_and_p_value[1][1]),
    )
    return corr_and_p_value


def findCorrPvalBasedAndNot(serie1, serie2, method):
    """Splitting function for getting correlation and p-values. Based on input it chooses either:
    1) p-value based methods or 2) non-p-value based methods
    Input: Two pandas series, and correlation method as a string
    Output: 2 x two-value tuple, for example if asym: ((corr12, p12), (coor21, p21)). For sym corr12=corr21 and p12=p21"""

    if sf.function_dict[method][1] == "no_pvalue":
        corr_and_pvalues = sf.calcCorrAndMimicP(serie1, serie2, method)
    elif sf.function_dict[method][1] == "with_pvalue":
        corr_and_pvalues = sf.calcCorrAndP(serie1, serie2, method)
        corr_and_pvalues = corr_and_pvalues, corr_and_pvalues
    return corr_and_pvalues


def findCorrSelectMethod(
    serie1,
    serie2,
    catcol,
    CI=0.1,
    method_cc="Cramer_V",
    method_cn="Omega",
    method_nn="Pear",
):
    if serie1.name == serie2.name:
        corr_and_pvalues = ((0, 1), (0, 1))
        return corr_and_pvalues
    else:
        type_serie1 = ["cat" if serie1.name in catcol else "num"][0]
        type_serie2 = ["cat" if serie2.name in catcol else "num"][0]
        corr_type = corrType(type_serie1, type_serie2)
        corr_method = corrMethodsExecute(corr_type, method_cc, method_cn, method_nn)
        corr_and_pvalues = findCorrPvalBasedAndNot(serie1, serie2, corr_method)

    # Lastly, we screen corr_values based on pvalues and restore them in the variable corr_and_pvalues
    corr_and_pvalues = screenCorrValuesBasedOnPvalue(corr_and_pvalues, CI)

    return corr_and_pvalues


def idLabel(
    cat_cols=None,
    num_cols=None,
    CI=None,
    method_cc=None,
    method_cn=None,
    method_nn=None,
):
    cat_cols = cat_cols
    num_cols = num_cols
    entry_id = f"CATCOLS: {cat_cols} - NUMCOLS: {num_cols} - CI: {CI} - METHOD_CC: {method_cc} - METHOD_CN: {method_cn} - METHOD_NN: {method_nn}"
    return entry_id


def correlation(
    dataset,
    catcols,
    numcols,
    CI=0.1,
    method_cc="Cramer_V",
    method_cn="Omega",
    method_nn="Pear",
):
    """Returns association strength matrix of any combination between numerical and categorical variables.
    Returns 2 matrices: one with corelation values and one with p-values"""
    # Arrange dataset so categoric columns are to the right
    dataset = dataset[catcols + numcols]
    columns = dataset.columns

    # create id_label
    entry_id = idLabel(
        cat_cols=catcols,
        num_cols=numcols,
        CI=CI,
        method_cc=method_cc,
        method_cn=method_cn,
        method_nn=method_nn,
    )

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
                CI=CI,
                method_cc=method_cc,
                method_cn=method_cn,
                method_nn=method_nn,
            )
            # Add calc values to each cell pairs in the matrices
            asso_matrix_corr.iloc[i][j] = corr_and_pvalues[0][0]
            asso_matrix_corr.iloc[j][i] = corr_and_pvalues[1][0]
            asso_matrix_p.iloc[i][j] = corr_and_pvalues[0][1]
            asso_matrix_p.iloc[j][i] = corr_and_pvalues[1][1]
    return asso_matrix_corr, asso_matrix_p, entry_id


# df_test = pd.read_excel(
#     "/home/jesper/Work/MLDA_app/MLDA/input_data/fortest_DF_shuf.xlsx"
# )
# # # Exercise
# catcols = [
#     "global_warm_risk",
#     "intimacy",
#     "President",
#     "gender",
#     "group",
#     "Color",
#     "num_police",
# ]
# numcols = ["X1", "weight", "X3", "Y1"]

# actual = correlation(
#     df_test,
#     catcols,
#     numcols,
#     CI=1,
#     method_cc="Asym",
#     method_cn="Omega",
#     method_nn="Spearmann",
# )

# print(actual[0])
# print()
# print(actual[1])
# print()
# print(actual[2])

# print("hey")
# data_input = pd.read_csv("/home/jesper/Work/macledan/input_files/data.csv")
# numcols = [
#     "Age",
#     "Overall",
#     "Potential",
#     "Crossing",
#     "Finishing",
#     "ShortPassing",
#     "Dribbling",
#     "LongPassing",
#     "BallControl",
#     "Acceleration",
#     "SprintSpeed",
#     "Agility",
#     "Stamina",
#     "Value",
#     "Wage",
# ]
# numcols = []
# catcols = ["Preferred Foot", "Position", "Body Type"]

# data = data_input[catcols + numcols]

# actual = correlation(data, catcols, numcols, CI=1, method_cc="Asym")

# print(actual)

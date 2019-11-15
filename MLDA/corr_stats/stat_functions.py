import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from statistics import mean, stdev
from sklearn.utils import shuffle
from sklearn.feature_selection import f_regression, mutual_info_regression
from scipy import stats
from statsmodels.formula.api import ols


def rescaling(serie):
    """Transform a feature/series to have a min of 0 and max of 1"""
    type_serieinput = type(serie)  # Uncomment while debugging
    try:
        name = serie.name  # store series name
        values = serie.values.reshape(-1, 1)  # convert to numpy array
        # Create scaler
        minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # Scale feature
        scaled = minmax_scaler.fit_transform(values)
        serie_scaled = pd.Series(np.ravel(scaled))
        serie_scaled.name = name
        type_serie_scaled = type(serie_scaled)  # Uncomment while debugging
        return serie_scaled
    except:
        return "Not scalable"


def cramers_v(x, y):
    """Effect size by Cramer's V. Formula: Cramer's V = Chi_sq/(n * (r_k_min - 1)), where r_k_min=min of the r-k dimension of the confusion_matrix, r=num_rows ,
    k=num_columns, n=grand total of observations
    Input: Two pandas series. 
    Output: tuple with: Cramer's V (scalar/float) and p-value (scalar/float)"""
    type_xinput, type_yinput = type(x), type(y)  # Uncomment while debugging
    confusion_matrix = pd.crosstab(x, y)
    chi_sq = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    r_k_min = min(r, k)
    nominator = chi_sq[0]
    denominator = n * (r_k_min - 1)
    # The if clause below is only for testing purpose
    if denominator==0:
        out = (np.nan, np.nan)
    else:
        cramer_v = (chi_sq[0] / (n * (r_k_min - 1))) ** 0.5
        out = cramer_v, chi_sq[1]  # out: cramer's V and p-value (tuple)
    type_out = type(out)  # Uncomment while debugging
    return out


def cramers_v_corr(x, y):
    """
    Input: Two pandas series. 
    Output: tuple with: Cramer's V corrected (scalar/float) and p-value (scalar/float)"""
    type_xinput, type_yinput = type(x), type(y)  # Uncomment while debugging
    confusion_matrix = pd.crosstab(x, y)
    chi_sq = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    chi_sq_corr = chi_sq[0] / n - (k - 1) * (r - 1) / (n - 1)
    chi_sq_corr = max(0, chi_sq_corr)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    r_k_corr_min = min(r_corr, k_corr)
    # The if clause below is only for testing purpose
    denominator=(r_k_corr_min - 1)
    if denominator==0:
        out = (np.nan, np.nan)
    else:
        cramer_v_corr = (chi_sq_corr / (r_k_corr_min - 1)) ** 0.5
        out = cramer_v_corr, chi_sq[1]  # cramer's V corrected and p-value
    type_out = type(out)  # Uncomment while debugging
    return out


def zero_replace(array):
    """Approved
    Part 0 of 3 - Helper function"""
    type_arrayinput = type(array)  # Uncomment while debugging
    for r in range(array.shape[0]):
        for c in range(array.shape[1]):
            if array[r][c] == 0:
                array[r][c] = 1
    type_arrayoutput = type(array)  # Uncomment while debugging
    return array


def calc_Uy(array):
    """Approved
    Part 1 of 3 - Uncertainty coefficient"""
    type_arrayinput = type(array)  # Uncomment while debugging
    Uy = 0
    n = array.sum()
    for col in range(array.shape[1]):
        f_dot_c = array[:, [col]].sum()
        Uy += (f_dot_c / n) * math.log10(f_dot_c / n)
    type_arrayoutput = type(array)  # Uncomment while debugging
    return -Uy


def calc_Uyx(array):
    """Approved
    Part 2 of 3 - Uncertainty coefficient"""
    type_arrayinput = type(array)  # Uncomment while debugging
    n = array.sum()
    Uyx = 0
    for r in range(array.shape[0]):
        f_r_dot = array[[r], :].sum()
        for c in range(array.shape[1]):
            f_rc = array[r][c]
            Uyx += (f_rc / n) * math.log10(f_rc / f_r_dot)
    type_arrayoutput = type(array)  # Uncomment while debugging
    return -Uyx


def u_y(y, x):
    """Approved
    Part 3a of 3 - Uncertainty coefficient (asymmetric)
    Input: Two pandas series. 
    Output: tuple with: Uncertainty coefficient (scalar/float) and p-value (scalar/float)
    """
    type_xinput, type_yinput = type(x), type(y)  # Uncomment while debugging
    array = pd.crosstab(y, x).values
    replace_zeroes = zero_replace(array)  # if cell value is zero we replace with 1
    Uy = calc_Uy(array)
    Uyx = calc_Uyx(array)
    out = (Uy - Uyx) / Uy, None  # uncertainty coeff, None
    type_out = type(out)  # Uncomment while debugging
    return out  # returns uncertainty coeff and None


def MI_cat(y, x):
    """Approved
    Part 3b of 3 - Mutual Information (symmetric)
    Input: Two pandas series. 
    Output: tuple with: Uncertainty coefficient (scalar/float) and p-value (scalar/float)
    """
    type_xinput, type_yinput = type(x), type(y)
    array = pd.crosstab(y, x).values
    replace_zeroes = zero_replace(array)  # if cell value is zero we replace with 1
    Uy = calc_Uy(array)
    Uyx = calc_Uyx(array)
    out = Uy - Uyx, None
    type_MI_cat_output = type(out)  # Uncomment while debugging
    return out  # returns tuple: Mutual information (MI) and None


def MI_num(x, y):
    """
    Input: Two pandas series. 
    Output: tuple with: Uncertainty coefficient (scalar/float) and None
    """
    type_xinput, type_yinput = type(x), type(y)  # Uncomment while debugging
    # convert to numpy array and reshape due to 1D data
    x_numpy = x.values.reshape(-1, 1)
    # Below: to get the scalar-value in out from the one-item list, we slice with:[0]
    out = mutual_info_regression(x_numpy, y.values)[0], None
    type_out = type(
        mutual_info_regression(x_numpy, y.values)
    )  # Uncomment while debugging
    return out


def spearmann(x, y):
    """
    Input: Two pandas series. 
    Output: tuple with: Spearmann correlation coefficient (scalar/float) and p-value
    """
    type_xinput, type_yinput = type(x), type(y)  # Uncomment while debugging
    out = stats.spearmanr(x, y)
    type_out = type(out)  # Uncomment while debugging
    return out


def pearson(x, y):
    """
    Input: Two pandas series. 
    Output: tuple with: Pearson's correlation coefficient (scalar/float) and p-value
    """
    type_xinput, type_yinput = type(x), type(y)  # Uncomment while debugging
    out = stats.pearsonr(x, y)
    type_out = type(out)  # Uncomment while debugging
    return out


def omega_ols(x, y):
    """Using linear regression where we want to carry out our ANOVA
    Input: Two pandas series. 
    Output: tuple with: Omega (scalar/float) and p-value (scalar/float)
    NOTE x: MUST BE CATEGORICAL, y: MUST BE NUMERICAL"""
    type_xinput, type_yinput = type(x), type(y)  # Uncomment while debugging
    DFB = len(pd.unique(x)) - 1
    if DFB <= 0:
        return (np.nan, np.nan)
    data = pd.concat([x, y], axis=1)
    x_name, y_name = x.name, y.name
    func_arg = y_name + "~C(" + x_name + ")"  # make string-input to ols below
    model = ols(func_arg, data=data).fit()
    # model = ols(x.to_numpy(), y.to_numpy())
    # Naming:
    # SS:Sum-of-Squares, B:Between, W:Within, T:Total, DF:Degrees-of-Freedom, mse:mean-square-error
    DFB, DFW, DFT = model.df_model, model.df_resid, (model.df_model + model.df_resid)
    SSB, SST = model.mse_model * DFB, model.mse_total * DFT
    mse_W = model.mse_resid
    omega_sq = (SSB - (DFB * mse_W)) / (SST + mse_W)
    if omega_sq < 0:
        omega_sq = 0
    out = (omega_sq) ** 0.5, model.f_pvalue  # omega and f_pvalue
    type_out = type(out)  # Uncomment while debugging
    return out


def removeNan(serie1, serie2):
    type_serie1_in, type_serie2_in = (
        type(serie1),
        type(serie2),
    )  # Uncomment while debugging
    df = pd.concat([serie1, serie2], axis=1)
    df_drop = df.dropna()
    out = df_drop.iloc[:, 0], df_drop.iloc[:, 1]
    type_out = type(out)  # Uncomment while debugging
    return out


function_dict = {
    "Omega": omega_ols,
    "Cramer_V": cramers_v,
    "Cramer_V_corr": cramers_v_corr,
    "Theils_u": u_y,
    "Uncer_coef": u_y,
    "Asym": u_y,
    "MI_cat": MI_cat,
    "MI_num": MI_num,
    "Spear": spearmann,
    "Spearmann": spearmann,
    "Pearsons": pearson,
    "Pear": pearson,
}


def returnListOfMethods(key):
    """A function that based on request returns a list of methods. Requests can be either 'with_pvalue' or 'no_pvalue'"""
    with_pvalue = ["Omega", "Cramer_V", "Spear", "Spearmann", "Pear", "Pearsons"]
    no_pvalue = ["Theils_u", "Uncer_coef", "Asym", "MI_cat", "MI_num"]
    dict_with_methods = {"with_pvalue": with_pvalue, "no_pvalue": no_pvalue}
    return dict_with_methods[key]

def mimicPvalueCalc(serie1, serie2, method):
    corr_val = function_dict[method](serie1, serie2)[0]
    mean_and_std = calcMeanAndStdDev(method, serie1, serie2)
    p_value = mean_and_std[1]/(corr_val-mean_and_std[0])

def calcMeanAndStdDev(method, serie1, serie2):
    """Input is a pandas serie. Calc mean and std of the correlation value based on 5 corr value estimations"""
    type_method_in, type_serie1_in, type_serie2_in = (
        type(method),
        type(serie1),
        type(serie2),
    )  # Uncomment while debugging
    corr_values = []
    serie2_name = serie2.name
    np.random.seed(0)
    for cycle in range(5):
        # first convert serie2 to numpy, then shuffle serie2, and lastly reconvert to pandas Series type
        serie2 = pd.Series(shuffle(serie2.to_numpy()), name=serie2_name)
        # value = function_dict[method](serie1, serie2)[0]
        corr_value = float(function_dict[method](serie1, serie2)[0])
        corr_values.append(corr_value)
    out = mean(corr_values), stdev(corr_values)
    type_out = type(out)  # Uncomment while debugging
    return out


def evalSignificance(method, serie1, serie2, CI=0.1):
    """Evaluates significance based on the argument values (p_val and std_val).
    Calc is based on p-value if they exist for the method used, else a value based on mean and std for shuffled corr-values is used.
    Input: method=choose string_name from function_dict, serie1/2=pandas series
    Output: corr_value (scalar/float) and p-value (scalar/float)"""

    corr_values = function_dict[method](serie1, serie2) # tuple
    
    # If non-p-value based method
    if method in returnListOfMethods("no_pvalue"):
        mean_and_std = calcMeanAndStdDev(method, serie1, serie2)
        if corr_values[0] > (mean_and_std[0] + mean_and_std[1] * (1/CI)):
            return corr_values[0]
        else:
            return "Corr is Insignificant"

    # If p-value based method
    elif method in returnListOfMethods("with_pvalue"):
        pvalue = corr_values[1]
        if pvalue < CI:
            return corr_values[0]
        else:
            return "p-value > CI"


# df_test = pd.read_excel("/home/jesper/Work/macledan/input_files/test_DF.xlsx")
# serie1, serie2 = df_test["global_warm_risk"], df_test["weight"]
# serie1, serie2 = removeNan(df_test["X3"], df_test["Y1"])
# print(calcMeanAndStdDev("MI_cat", serie1, serie2))
# print(MI_num(serie1, serie2))
# print(evalSignificance("MI_num", serie1, serie2, std_val=15))
# print(pd.crosstab(serie1, serie2))

# # print(evalSignificance("Spearmann", df_test["X1"], df_test["Y1"]))
# print(df_test)
# serie1, serie2 = removeNan(df_test["X1"], df_test["Y1"])

# print(spearmann(serie1, serie2))

# serie1, serie2 = df_test["X1"], df_test["X3"]
# serie1, serie2 = df_test["intimacy"], df_test["num_police"]

# serie1, serie2 = removeNan(serie1, serie2)
# print(calcMeanAndStd("Asym", serie1, serie2))
# print(evalSignificance("Asym", serie1, serie2, CI=0.1))
# print(omega_ols(serie1, serie2))
# print(type(serie1.to_numpy()))
# serie2_name = serie2.name
# serie2 = serie2.to_numpy()
# serie2 = pd.Series(serie2, name=serie2_name)
# print(serie1.name, serie2.name)
# print(omega_ols(serie1, serie2))
# Setup
# desired = 0.006285953589698483
# # Exercise
# np.random.seed(0)
# desired = "Corr is Insignificant"

# # Exercise
# serie1, serie2 = df_test["global_warm_risk"], df_test["gender"]
# actual = evalSignificance("MI_cat", serie1, serie2, std_val=10)
# print(actual)

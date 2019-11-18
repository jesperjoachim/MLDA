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
    if denominator == 0:
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
    denominator = r_k_corr_min - 1
    if denominator == 0:
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
    Output: tuple with: Uncertainty coefficient (scalar/float)
    """
    type_xinput, type_yinput = type(x), type(y)  # Uncomment while debugging
    array = pd.crosstab(y, x).values
    replace_zeroes = zero_replace(array)  # if cell value is zero we replace with 1
    Uy = calc_Uy(array)
    Uyx = calc_Uyx(array)
    out = (Uy - Uyx) / Uy  # uncertainty coeff
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
    out = Uy - Uyx
    type_MI_cat_output = type(out)  # Uncomment while debugging
    return out  # returns single value ('numpy.float64'): Mutual information


def MI_num(x, y):
    """
    Input: Two pandas series. 
    Output: tuple with: Uncertainty coefficient (scalar/float) and None
    """
    type_xinput, type_yinput = type(x), type(y)  # Uncomment while debugging
    # convert to numpy array and reshape due to 1D data
    x_numpy = x.values.reshape(-1, 1)
    # Below: to get the scalar-value in out from the one-item list, we slice with:[0]
    out = mutual_info_regression(x_numpy, y.values)[0]
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
    "Omega": [omega_ols, "with_pvalue"],
    "Cramer_V": [cramers_v, "with_pvalue"],
    "Cramer_V_corr": [cramers_v_corr, "with_pvalue"],
    "Theils_u": [u_y, "no_pvalue"],
    "Uncer_coef": [u_y, "no_pvalue"],
    "Asym": [u_y, "no_pvalue"],
    "MI_cat": [MI_cat, "no_pvalue"],
    "MI_num": [MI_num, "no_pvalue"],
    "Spear": [spearmann, "with_pvalue"],
    "Spearmann": [spearmann, "with_pvalue"],
    "Pearsons": [pearson, "with_pvalue"],
    "Pear": [pearson, "with_pvalue"],
}


def returnListOfMethods(key):
    """A function that based on request returns a list of methods. Requests can be either 'with_pvalue' or 'no_pvalue'"""
    with_pvalue = ["Omega", "Cramer_V", "Spear", "Spearmann", "Pear", "Pearsons"]
    no_pvalue = ["Theils_u", "Uncer_coef", "Asym", "MI_cat", "MI_num"]
    dict_with_methods = {"with_pvalue": with_pvalue, "no_pvalue": no_pvalue}
    return dict_with_methods[key]


def calcCorrNonP(serie1, serie2, method):
    """Function that calls calculation based on method. It handles entropy based methods - i.e non-p-value based methods. 
    Discriminate between asymmetric and symmetric methods.
    Input: Two pandas series, and correlation method
    Output: single value (symmetric) or two-value tuple (asynmmetric), for example if asymmetric: (corr12, coor21). 
    For symmetric cases: only corr"""
    # Input types:
    input_serie1, input_serie2, input_method = (
        type(serie1),
        type(serie2),
        type(method),
    )  # Uncomment while debugging
    # If asym:
    if function_dict[method][0] == u_y:
        corr_values = (
            function_dict[method][0](serie1, serie2),
            function_dict[method][0](serie2, serie1),
        )
    # If symmetric
    else:
        corr_values = function_dict[method][0](serie1, serie2)
    return corr_values


def calcCorrAndP(serie1, serie2, method):
    """Function that calls calculation based on method. It handles conventionel statistical methods with p-values.
    Symmetric output only.
    Input: Two pandas series, and correlation method
    Output: Two float values (tuple): corr and p-value."""
    corr_and_pvalue = function_dict[method][0](serie1, serie2)
    return corr_and_pvalue


def calcCorrAndMimicP(serie1, serie2, method):
    """Splitting function for calculating correlation and p-value-like calculations for entropy-based statistics. 
    Input: Two pandas series, and correlation method
    Output: double two-value tuple, for example if asymmetric: ((corr12, p12), (coor21, p21)). For symmetric corr12=corr21 
    and p12=p21."""
    # Input types:
    input_serie1, input_serie2, input_method = (
        type(serie1),
        type(serie2),
        type(method),
    )  # Uncomment while debugging

    # corr_values returns single value (sym) or two-value tuple (asym)
    corr_values = calcCorrNonP(serie1, serie2, method)

    # If symmetric
    if isinstance(corr_values, float):
        # mean_and_std12_and21 returns a two-value tuple
        mean_and_stdev = calcMeanAndStdDev(serie1, serie2, method)
        mimic_p = mimicPvalueCalc(mean_and_stdev, corr_values)
        corr_and_p = corr_values, mimic_p
        return corr_and_p, corr_and_p  # return duplicate due two data structure

    # Asym
    elif len(corr_values) == 2:
        # mean_and_std12_and21 returns a 2 x two-value tuple
        mean_and_stdev12_and21 = (
            calcMeanAndStdDev(serie1, serie2, method),
            calcMeanAndStdDev(serie2, serie1, method),
        )
        # To calc p-value p12 we pass in the first part of the double tuple calc above and the first part of the
        # corr_values tuple above. For the p21 it is the last part of the tuples.
        mimic_p12_andp21 = (
            mimicPvalueCalc(mean_and_stdev12_and21[0], corr_values[0]),
            mimicPvalueCalc(mean_and_stdev12_and21[1], corr_values[1]),
        )
        corr12_and_p12, corr21_and_p21 = (
            (corr_values[0], mimic_p12_andp21[0]),
            (corr_values[1], mimic_p12_andp21[1]),
        )
        return corr12_and_p12, corr21_and_p21


def mimicPvalueCalc(mean_and_stdev, corr_value):
    """Formula: p = stdev/(corr_value-mean), where corr_value: the actual corr value, stdev and mean: the standard deviation 
    and mean of the corr-value for 5 shuffled serie2 rows.
    Input: two-value tuple (mean_and_stdev) and float64 (corr_value)
    Output: float64"""
    input_mean_stdev, input_corr_val = (
        type(mean_and_stdev),
        type(corr_value),
    )  # Uncomment while debugging
    # input to the below calc of p-value is a tuple (mean_and_stdev) and a single value (corr_value)
    p_value = mean_and_stdev[1] / (corr_value - mean_and_stdev[0])
    return p_value


def calcMeanAndStdDev(serie1, serie2, method):
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
        # value = function_dict[method][0](serie1, serie2)
        corr_value = float(function_dict[method][0](serie1, serie2))
        corr_values.append(corr_value)
    out = mean(corr_values), stdev(corr_values)
    type_out = type(out)  # Uncomment while debugging
    return out


def evalSignificance(method, serie1, serie2, CI=0.1):
    """Evaluates significance based on the argument values (p_val and std_val).
    Calc is based on p-value if they exist for the method used, else a value based on mean and std for shuffled corr-values is used.
    Input: method=choose string_name from function_dict, serie1/2=pandas series
    Output: corr_value (scalar/float) and p-value (scalar/float)"""

    corr_values = function_dict[method][0](
        serie1, serie2
    )  # returns single value or two-value tuple

    # If non-p-value based method
    if method in returnListOfMethods("no_pvalue"):
        mean_and_std = calcMeanAndStdDev(method, serie1, serie2)
        if corr_values[0] > (mean_and_std[0] + mean_and_std[1] * (1 / CI)):
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

# serie1, serie2 = df_test["X1"], df_test["Y1"]
# serie1, serie2 = removeNan(serie1, serie2)
# actual = calcCorrAndP(serie1, serie2, "Spear")

# # Verify
# print(type(actual).__name__)
# print(dir(actual))
# print(len(actual))

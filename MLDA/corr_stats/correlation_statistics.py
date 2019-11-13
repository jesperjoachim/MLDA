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

# %run ~/Work/macledan/packages_import.ipynb # importing package dependencies + functions

# +
# seaborn settings
sns.set(style="ticks")
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
flatui = sns.color_palette(flatui)

data_input = pd.read_csv("data.csv")

data_input = spacesToUnderscore(data_input) # Change the spaces in column names to underscores

numcols = [
    'Age',
 'Overall',
 'Potential',
'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility',  'Stamina',
 'Value','Wage']
catcols = ['Name','Club','Nationality','Preferred_Foot','Position','Body_Type']

# Subset the columns
data = data_input[numcols + catcols]

# data_input.head()
# data.info()
data.head(5)
    
# -

# ### Preprocessing the Wage and Value 

# +
def wage_split(x):
    try:
        if str(x).isdigit():
            return x
        else: 
            return int(x.split("K")[0][1:])
    except:
        return x

def value_split(x):
    try:
        if str(x).isdigit():
            return x
        elif 'K' in x:
            return float(x.split('K')[0][1:])/1000
        elif 'M' in x:
            return float(x.split('M')[0][1:])
    except:
        return x

data['Wage'] = data['Wage'].apply(lambda x: wage_split(x))
data['Value'] = data['Value'].apply(lambda x: value_split(x))

data.head()
data.tail()
# -

# ### Numerical variables Correlation with Heatmap

# +
data_num_corr = data.corr(method='spearman') # default is pearson's, third option is 'kendall'
type(data_num_corr)
data_num_corr.head()
g = sns.heatmap(data_num_corr, vmax=.6, center=0, square=True,
               linewidths=.5, cbar_kws={'shrink': .5}, annot=True,
               fmt='.2f', cmap='Greens')
# sns.despine()
g.figure.set_size_inches(14, 10)

plt.show() # NOTE THIS ONLY SHOWS THE NUMERIC CORRELATIONS

# -

# ### Categorical and Numerical variables Correlation

# ### Functions for  variables Correlation

# + {"code_folding": [5]}
def chi_sq(x, y):
    """Approved"""
    confusion_matrix = pd.crosstab(x, y)
    chi_sq = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    r_k_min = min(r, k)
    nominator = chi_sq[0]
    denominator = (n*(r_k_min-1))
    cramer_v = (chi_sq[0]/(n*(r_k_min-1)))**.5
    return chi_sq[0]
    
def cramers_v(x, y):
    """Approved"""
    confusion_matrix = pd.crosstab(x, y)
    chi_sq = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    r_k_min = min(r, k)
    nominator = chi_sq[0]
    denominator = (n*(r_k_min-1))
    cramer_v = (chi_sq[0]/(n*(r_k_min-1)))**.5
    return cramer_v
    
def cramers_v_corr(x, y):
    """Approved"""
    confusion_matrix = pd.crosstab(x, y)
    chi_sq = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    chi_sq_corr = max(0, chi_sq[0]/n-(k-1)*(r-1)/(n-1))
    k_corr = k-((k-1)**2)/(n-1)
    r_corr = r-((r-1)**2)/(n-1)
    r_k_corr_min = min(r_corr, k_corr)
    cramer_v_corr = (chi_sq_corr/(r_k_corr_min-1))**.5
    return cramer_v_corr

def zero_replace(array):
    """Approved
    Part 0 of 3 - Helper function"""
    for r in range(array.shape[0]):
        for c in range(array.shape[1]):
            if array[r][c] == 0:
                array[r][c] = 1
    return array

def calc_Uy(array):
    """Approved
    Part 1 of 3 - Uncertainty coefficient"""
    Uy = 0
    n = array.sum()
    for col in range(array.shape[1]):
        f_dot_c = array[:,[col]].sum()
        Uy += (f_dot_c/n)*math.log10(f_dot_c/n)
    return -Uy

def calc_Uyx(array):
    """Approved
    Part 2 of 3 - Uncertainty coefficient"""
    n = array.sum()
    Uyx = 0
    for r in range(array.shape[0]):
        f_r_dot = array[[r],:].sum()
        for c in range(array.shape[1]):
            f_rc = array[r][c]
            Uyx += (f_rc/n)*math.log10(f_rc/f_r_dot)
    return -Uyx

def u_y(y, x):
    """Approved
    Part 3a of 3 - Uncertainty coefficient (asymmetric)"""
    array = pd.crosstab(y,x).values
    replace_zeroes = zero_replace(array) # if cell value is zero we replace with 1
    Uy = calc_Uy(array)
    Uyx = calc_Uyx(array)
    return (Uy - Uyx)/Uy # returns uncertainty coeff

def MI_cat(y, x):
    """Approved
    Part 3b of 3 - Mutual Information (symmetric)"""
    array = pd.crosstab(y,x).values
    replace_zeroes = zero_replace(array) # if cell value is zero we replace with 1
    Uy = calc_Uy(array)
    Uyx = calc_Uyx(array)
    return Uy - Uyx # returns Mutual information (MI)

def MI_num(x, y):
    x_datatype = x.values.reshape(-1, 1)
    return mutual_info_regression(x_datatype, y)[0]

def spearmann(x, y, CI=0.1):
    # Check confidens interval (CI)
    if stats.spearmanr(x, y)[1] > CI:
        return 0
    else:
        return stats.spearmanr(x, y)[0]
    
def pearson(x, y, CI=0.1):
    # Check confidens interval (CI)
    if stats.pearsonr(x, y)[1] > CI:
        return 0
    else:
        return stats.pearsonr(x, y)[0]

def omega_ols(x, y):
    """Using linear regression where we want to carry out our ANOVA
    NOTE x: MUST BE CATEGORICAL, y: MUST BE NUMERICAL"""
    data = pd.concat([x, y], axis=1)
    x_name, y_name = x.name, y.name
#     data = data[[x.name, y.name]]
    func_arg = y_name + "~C(" + x_name + ')' # make string-input to ols below
    model = ols(func_arg, data=data).fit()
    # Naming: 
    # SS:Sum-of-Squares, B:Between, W:Within, T:Total, DF:Degrees-of-Freedom, mse:mean-square-error
    DFB, DFW, DFT = model.df_model, model.df_resid, (model.df_model + model.df_resid)
    SSB, SST = model.mse_model*DFB, model.mse_total*DFT
    mse_W = model.mse_resid
    omega_sq = (SSB-(DFB*mse_W))/(SST+mse_W)
    return (omega_sq)**.5

def omega(x, y):
    """Same as omega_ols but calculated manually
    NOTE x: MUST BE CATEGORICAL, y: MUST BE NUMERICAL"""
    N_grps = pd.unique(x.values)
    data = pd.concat([x, y], axis=1)
    x_name, y_name = x.name, y.name
    dict_data = {grp:y[x == grp] for grp in N_grps}
    # Naming: 
    # SS:Sum-of-Squares, B:Between, W:Within, T:Total, DF:Degrees-of-Freedom, mse:mean-square-error
    DFT, DFB = len(x)-1, len(pd.unique(x))-1
    DFW = DFT - DFB
    SST = sum([(yi-y.mean())**2 for yi in y])
    # SSB = Sum of (Ni*(Yav_i-Y_gm)**2) for all groups
    SSB = sum([len(dict_data[key])*(dict_data[key].mean()-y.mean())**2 for key in dict_data])
    SSW = SST - SSB
    omega_sq = (SSB - (DFB*SSW/DFW))/(SST+SSW/DFW)
    return omega_sq**.5

function_dict = {
    "Omega_ols": omega_ols,
    "Omega": omega,
    "Chi_sq": chi_sq,
    "Cramer_V": cramers_v,
    "Cramer_V_corr": cramers_v_corr,
    "Theils_u": u_y,
    "Uncer_coef": u_y,
    "Asym": u_y,
    "MI_cat": MI_cat,
    "MI_num": MI_num,
    "Spear": spearmann,
    'Pear': pearson
    }

def corrType(type1, type2):
    if type1 == 'cat' and type2== 'cat':
        return 'catcat'
    elif type1 == 'num' and type2== 'num':
        return 'numnum'
    else:
        return 'catnum'

def corrMethodsExecute(corr_type, method_cc, method_cn, method_nn):
    dict_methods = {
        'catcat': method_cc,
        'catnum': method_cn,
        'numnum': method_nn,
    }
    return dict_methods[corr_type]

def removeNan(serie1, serie2):
    try:
        df = pd.concat([serie1, serie2], axis=1)
        df_drop = df.dropna()
        logger.debug(f'Return 2 nan-free series, {df_drop.iloc[:,0].name}, {df_drop.iloc[:,1].name}')
        return df_drop.iloc[:,0], df_drop.iloc[:,1]
    except:
        logger.warning(f'Could not return, {serie1.name}, {serie2.name}')
        

def corrValue(serie1, serie2, methods):
    """Returns the asym values of serie1/2 (i.e 2 diff values), or the max values (same)
    for the symmetric case"""   
    try:
        corr_values = []
        if 'Asym' in methods:
            # asymmetric matrix
            method = methods[0]
            logger.debug(f'method: {method}')
            logger.debug(f'function_dict[method](serie1, serie2): {function_dict[method](serie1, serie2)}, function_dict[method](serie2, serie1):{function_dict[method](serie2, serie1)}')
            corr_values.extend((function_dict[method](serie1, serie2), function_dict[method](serie2, serie1)))
        elif methods:
            # symmetric matrix
            value = []
            for method in methods:
                value.append(function_dict[method](serie1, serie2))
            value_max = max(value)
            logger.debug(f'value and value_max: {value}, {value_max}')
            corr_values.extend((value_max, value_max))
        else:
            return 'error in methods'
        logger.debug(f'corr_values returned: {corr_values}')
        return corr_values
    except Exception as ex:
        logger.debug(f'Unable to finish corrValue, due to: {ex}')
            
def findCorr(serie1, serie2, method_cc = ['Asym'], method_cn = ['Omega_ols'], method_nn=['Spear', 'Pear', 'MI_num']):
    try:
        logger.debug(f'Init findCorr with serie1: {serie1.name} and serie2: {serie2.name}')
        if serie1.name == serie2.name:
            corr_value = (1, 1)
        else:
            type_serie1 = ['cat' if serie1.name in catcol else 'num'][0]
            type_serie2 = ['cat' if serie2.name in catcol else 'num'][0]
            corr_type = corrType(type_serie1, type_serie2) 
            logger.debug(f'corr_type: {corr_type}')

            corr_methods = corrMethodsExecute(corr_type, method_cc, method_cn, method_nn)
            logger.debug(f'corr_methods: {corr_methods}')

            corr_value = corrValue(serie1, serie2, corr_methods)
            logger.debug(f'corr_value: {corr_value}')
        return corr_value
    except Exception as ex:
        logger.debug(f'Unable to finish findCorr, due to: {ex}')


# + {"hide_input": true}
x1 = [2,3,4,7,9, np.nan]
x2 = [4,6,6,7,14, np.nan]
serie1 = pd.Series(x1, name='x1')
serie2 = pd.Series(x2, name='x2')
df12=pd.concat([serie1, serie2], axis=1)
# mi = MI_num(serie1, serie2)
# mi
# x1

# + {"hide_input": false, "cell_type": "markdown"}
# #### Hided stuff below here

# + {"hide_input": true}
# # data.head(2)
# cat1_dum = pd.get_dummies(data['Preferred Foot'], drop_first=True)
# cat2_dum = pd.get_dummies(data['Position'], drop_first=True)
# # cat2_dum.head()
# cat1_2_dum_concat = pd.concat([cat1_dum, cat2_dum], axis=1)
# cat1_2_dum_concat.head()

# columns = cat1_2_dum_concat.columns
# print('columns', columns[[0, 1, 7, 17]])
# # cat1_2_dum_concat.loc[0:10,columns[[0, 1, 7, 17]]]
# confusion_matrix = pd.crosstab(data['Preferred Foot'], data['Position'])
# print('confusion_matrix.shape', confusion_matrix.shape)
# right_foot = cat1_2_dum_concat.loc[0:100, [columns[0]]]
# CB_LB_RB = cat1_2_dum_concat.loc[0:100,columns[[1, 7, 17]]]
# CB_LB_RB
# # data[columns[0]]
# right_foot.shape
# CB_LB_RB.shape
# # confusion_matrix = pd.crosstab(right_foot, CB_LB_RB)



# def association_cat(dataset):
#     for i in range(0, len(columns)):
#         for j in range(i, len(columns)):
#             cramers = (energy[columns[i]], energy[columns[j]])
#             corr_p, p_p = stats.pearsonr(energy[columns[i]], energy[columns[j]])
#             # if both correlations have p-values above 10% CI, add: 'p>10%'
#             if p_s > .1 and p_p > .1:
#                 df_corr[columns[i]][columns[j]], df_corr[columns[j]][columns[i]] = 0, 0
#             # if spearman correlation have p-value above 10% CI and pearson's do not, add: corr_p
#             elif p_s > .1 and p_p < .1:
#                 df_corr[columns[i]][columns[j]], df_corr[columns[j]][columns[i]] = corr_p, corr_p
#             # if spearman correlation < pearson's
#             elif p_s < p_p:
#                 df_corr[columns[i]][columns[j]], df_corr[columns[j]][columns[i]] = corr_s, corr_s
#             # if spearman correlation > pearson's
#             elif p_s > p_p:
#                 df_corr[columns[i]][columns[j]], df_corr[columns[j]][columns[i]] = corr_p, corr_p
#             else:
#                 df_corr[columns[i]][columns[j]], df_corr[columns[j]][columns[i]] = corr_s, corr_s

# df_corr

# # df_corr[df_corr.columns]
# df_corr = df_corr[df_corr.columns].astype(float)
# g = sns.heatmap(df_corr, vmax=.3, vmin=-.3, center=0, square=True,
#                linewidths=.5, cbar_kws={'shrink': .5}, annot=True,
#                fmt='.2f', cmap='coolwarm')
# # sns.despine()
# g.figure.set_size_inches(14, 10)

# plt.show()

# + {"hide_input": true}
catcol = ['Preferred_Foot','Body_Type', 'Position'] # 
numcol = ['Age', 'Value', 'Wage', 'BallControl', 'SprintSpeed']
data = data[catcol+numcol]
data_nan = data[data.isna().any(axis=1)]
data_nan.tail(70);
# for column in data.columns:
#     data[column].isna().sum() 

# + {"hide_input": true}
catcol = ['Preferred_Foot','Body_Type', 'Position'] # 
numcol = ['Age', 'Value', 'Wage', 'BallControl', 'SprintSpeed']
dataset = data[catcol+numcol]
serie1, serie2 = dataset['Value'], dataset['Value']
df = pd.concat([serie1, serie2], axis=1)
df_drop = df.dropna()
type(df_drop)
type(df_drop[serie1.name])#, df_drop[serie2.name]
df_drop
type(df_drop.iloc[0])
type(df_drop.iloc[1]);

# + {"hide_input": true}
frame = pd.DataFrame(np.arange(12.).reshape((4,3)), columns=list('bde'), index=['utah', 'ohio', 'texas', 'oregon'])
series = frame.iloc[:,1]
frame
# series.columns[0]
type(series)
series.name

def removeNaN(serie1, serie2):
    try:
        df = pd.concat([serie1, serie2], axis=1)
        df_drop = df.dropna()
        logger.debug(f'Return 2 nan-free series, {df_drop.iloc[0].name}, {df_drop.iloc[1].name}')
        return df_drop.iloc[0], df_drop.iloc[1]
    except:
        logger.warning(f'Could not return, {serie1.name}, {serie2.name}')


# +
catcol = ['Preferred_Foot','Body_Type', 'Position'] # 
numcol = ['Age', 'Value', 'Wage', 'BallControl', 'SprintSpeed']
dataset = data[catcol+numcol]

def correlation(dataset, catcols, numcols):
    """Returns association strength matrix of any combination between numerical and categorical variables"""
    # we choose which method to use for calc association strength
    columns = dataset.columns
    corr_matrix = pd.DataFrame(index=columns, columns=columns) # constructing a square matrix
    logger.debug('Created df-matrix for correlations')
    #     looping through each row and columns
#     print(columns)
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            logger.debug('Looping through the rows and columns')
            serie1, serie2 = removeNan(dataset[columns[i]],dataset[columns[j]])
            logger.debug(f'Calls findCorr function for {dataset[columns[i]].name} and {dataset[columns[j]].name}')
            corr_values = findCorr(serie1, serie2)
            logger.debug(f'Add these values: {corr_values[0]}, {corr_values[0]} to cell {[i,j]}, {[j,i]}')
            ########HERTIL!!! checked_corr_values = removeNanTuple(corr_values)
            corr_matrix.iloc[i][j] = corr_values[0]
            corr_matrix.iloc[j][i] = corr_values[1]
    return corr_matrix
correlation = correlation(dataset, catcol, numcol);


# +
correlation = correlation.fillna(0)
correlation

g = sns.heatmap(correlation, vmax=.3, center=0, square=True,
               linewidths=.5, cbar_kws={'shrink': .5}, annot=True,
               fmt='.2f', cmap='Greens')
# sns.despine()
g.figure.set_size_inches(14, 10)

# Due to bug in matplotlib 3.1.1
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)

# place xticks at the top
g.xaxis.set_ticks_position('top')
# rotate the top text
g.set_xticklabels(g.get_yticklabels(), rotation = 60)
plt.show();

# +
sel_dataset = data[selected_catcol+selected_numcols]
selected_catcol = ['Preferred Foot','Position','Body Type']
selected_numcols = ['Age', 'Value', 'Wage', 'BallControl', 'SprintSpeed']

cat_cols, num_cols = selected_catcol, selected_numcols
asso_strength= pd.DataFrame(index=cat_cols, columns=num_cols) # constructing a square matrix
num_cols[1]
cat_cols[0]
print(asso_strength.iloc[0][1])

# +
selected_catcol = ['Preferred_Foot','Position','Body_Type']
dataset_cat = data[selected_catcol]
# dataset_cat
columns = dataset_cat.columns
dataset_cat[columns[0]]
index[0][columns[1]

def corr_cat(dataset_cat, method='Theils_u'):
    """Function for finding correlation/association strength between categorcal variables"""
    # we choose which method to use for calc association strength
    asso_method = function_dict[method] # default is Theils_u
    dataset = convert(dataset_cat, 'dataframe')
    columns = dataset.columns
    asso_strength= pd.DataFrame(index=columns, columns=columns) # constructing a square matrix, index=columns
    # looping through each row and columns
    for i in range(len(index)): # downwards/rows
        for j in range(i, len(columns)): # sidewards/columns
            if i == j:
                asso_strength[index[i]][columns[j]] = 1 # diagonal
            else:
                find_asso = find_asso(index[i]][columns[j])
                asso_strength[index[i]][columns[j]] = asso_method(dataset[columns[i]],dataset[columns[j]])
                asso_strength[index[j]][columns[i]] = asso_method(dataset[columns[j]],dataset[columns[i]])
    return method, asso_strength

# correlation = corr_cat(dataset_cat=dataset_cat, method='Theils_u')
# print(correlation[0],'\n','\n', correlation[1])


# +
import scipy.stats as ss
from collections import Counter
import math 
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

def convert(data, to):
    converted = None
    if to == 'array':
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == 'list':
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == 'dataframe':
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data),to))
    else:
        return converted
    
def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    :param x: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :param y: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :return: float
    """
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta

def associations(dataset, nominal_columns=None, mark_columns=False, theil_u=False, plot=True,
                          return_results = False, **kwargs):
    """
    Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
    continuous features using:
     - Pearson's R for continuous-continuous cases
     - Correlation Ratio for categorical-continuous cases
     - Cramer's V or Theil's U for categorical-categorical cases
    :param dataset: NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    :param nominal_columns: string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
        columns are categorical, or None (default) to state none are categorical
    :param mark_columns: Boolean (default: False)
        if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
        continuous), as provided by nominal_columns
    :param theil_u: Boolean (default: False)
        In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
    :param plot: Boolean (default: True)
        If True, plot a heat-map of the correlation matrix
    :param return_results: Boolean (default: False)
        If True, the function will return a Pandas DataFrame of the computed associations
    :param kwargs:
        Arguments to be passed to used function and methods
    :return: Pandas DataFrame
        A DataFrame of the correlation/strength-of-association between all features
    """

    dataset = convert(dataset, 'dataframe')
    columns = dataset.columns
    if nominal_columns is None:
        nominal_columns = list()
    elif nominal_columns == 'all':
        nominal_columns = columns
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0,len(columns)):
        for j in range(i,len(columns)):
            if i == j:
                corr[columns[i]][columns[j]] = 1.0
            else:
                if columns[i] in nominal_columns:
                    if columns[j] in nominal_columns:
                        if theil_u:
                            corr[columns[j]][columns[i]] = theils_u(dataset[columns[i]],dataset[columns[j]])
                            corr[columns[i]][columns[j]] = theils_u(dataset[columns[j]],dataset[columns[i]])
                        else:
                            cell = cramers_v(dataset[columns[i]],dataset[columns[j]])
                            corr[columns[i]][columns[j]] = cell
                            corr[columns[j]][columns[i]] = cell
                    else:
                        cell = correlation_ratio(dataset[columns[i]], dataset[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                else:
                    if columns[j] in nominal_columns:
                        cell = correlation_ratio(dataset[columns[j]], dataset[columns[i]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                    else:
                        cell, _ = ss.pearsonr(dataset[columns[i]], dataset[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = ['{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col) for col in columns]
        corr.columns = marked_columns
        corr.index = marked_columns
    if plot:
        plt.figure(figsize=(20,20))#kwargs.get('figsize',None))
        sns.heatmap(corr, annot=kwargs.get('annot',True), fmt=kwargs.get('fmt','.2f'), cmap='coolwarm')
        plt.show()
    if return_results:
        return corr
player_df = player_df.fillna(0)
results = associations(player_df,nominal_columns=catcols,return_results=True)

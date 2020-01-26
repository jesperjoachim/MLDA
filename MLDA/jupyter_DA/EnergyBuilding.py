# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + hide_input=false
from MLDA.imports.DA_modules import *

# On/off plots
plots = 'on' # choose 'on' or 'off'
# -

# # 0. Loading Data, First Look and Choose Categorical/Numerical columns

# +
# read the data
data_input = pd.read_excel('/home/jesper/Work/MLDA_app/MLDA/input_data/ENB2012_data.xlsx')
# data_input = pd.read_csv('/home/jesper/Work/MLDA_app/MLDA/input_data/Outlet_sales.csv')
data_input.head()

# column names
for column in data_input.columns:
    print(column)
    
Var_names = dict(X1='relative_compactness', X2='surface_area', X3='wall_area', X4='roof_area', X5='overall_height', X6='orientation', X7='glazing_area', X8='glazing_area_distribution', Y1='heating_load', Y2='cooling_load')
Var_names_short = dict(X1='rel_compact', X2='surf_area', X3='wall_area', X4='roof_area', X5='height', X6='orientation', X7='glazing_area', X8='glaz_area_distrib', Y1='heat_load', Y2='cool_load')

# If no 'column names' or 'wrong names' in data_input, changes these

data_input.columns = [Var_names_short[key] for key in Var_names_short]
data_input.head()

# -

# ## 0.1 Choosing the relevant variables

data_input.head() # Columns to choose from
choose_var = [0,1,2,3,4,5,6,7,8,9] # First column: 0
data = data_input.iloc[:, choose_var]
data.head(15)

# + [markdown] hide_input=true
# ## 0.2 Separating data in cat/num classes and x/y classes, and a combination of these

# + hide_input=false
# Pick column numbers from data, NOT from data_input, first column is 0 (zero)
categoric_x = []
numeric_x = [0,1,2,3,4,5,6,7]
categoric_y = []
numeric_y = [8,9]

################ Calc below ###############
input_classes = {} # init a dict
x = categoric_x + numeric_x
y = categoric_y + numeric_y
categoric = categoric_x + categoric_y
numeric = numeric_x + numeric_y

# Adding to the dict
input_classes['cx'] = categoric_x
input_classes['nx'] = numeric_x
input_classes['x'] = x
input_classes['cy'] = categoric_y
input_classes['ny'] = numeric_y
input_classes['y'] = y
input_classes['c'] = categoric
input_classes['n'] = numeric
categoric

# + [markdown] hide_input=false
# ### 0.2.1 Convert the categorical data dtype to 'category' 
# -

for entry in categoric:
    data.iloc[:, entry] = data.iloc[:, entry].astype("category")

# +
# Checking that the data are of the correct dtype
# data['Item_Fat_Content'].dtype.name

# + [markdown] hide_input=false
# ### 0.2.2 Creating new names based on the type of the variable - i.e is it x or y and is it categorical or numerical

# + hide_input=true
# Store the original column names and order
col_orig = data.columns

# Creating new names based on x, y and categoric, numeric
col_xc = ['xc'+str(num+1) for num, item in enumerate(input_classes['cx'])] # the item part is not used
col_xn = ['xn'+str(num+1) for num, item in enumerate(input_classes['nx'])] 
col_yc = ['yc'+str(num+1) for num, item in enumerate(input_classes['cy'])] 
col_yn = ['yn'+str(num+1) for num, item in enumerate(input_classes['ny'])] 
col_xy = col_xc + col_xn + col_yc + col_yn

print('col_orig:' ,col_orig)
print('x+y:', x+y)
print('col_xy:' ,col_xy)

# Transferring the right (original) ordering of the variables to the new names (x, y)
order_orig =[]

for order, item in enumerate(x+y):
    print(order, item)

# The Approach for the loop below:
# for each item in our new-name-list (col_xy) we find the original place/order of that item
# by looping through each number in the 'x+y'-list. The number in the 'x+y'-list is the 
# original order of the items in the new-name-list (col_xy). By picking that number - for example
# say 0, which say is in the 5'th place, we know by looking in col_xy at 5'th place that this item
# was original the first one. We then append this item as the first one to the list: order_orig.
# Then we pick number two, which is 1, etc

# we loop through each number from 0 to total number of columns, len(x+y)
for num in range(len(x+y)):
    # So we find 0 first...
    for order, item in enumerate(x+y):
        # pick 0 in list: x+y: let's say it is in the 5th place
        if item == num:
            # then in col_xy in the 5th place we know that this item was orig the first one 
            order_orig.append(col_xy[order])

# we restore the original order by replacing col_xy with order_orig
col_xy = order_orig

data
# -

# # 1. Data Munging

# ## 1.1 Getting an Overview
# __Conclusion based on cell below:__ <br>
# 1) write... <br>
# 2) write... <br>

# +
# Check the percentage of missing values (NaN values) for each parameter
data.isnull().sum()/len(data)*100
# Conclusion: Item_weight and Outlet_size has missing values: 17% and 28 %, respevtively

data.describe()
data.info()

for column in data.columns:
    data[column].value_counts()
# -

# ## 1.2 Munging the relevant columns

# +
# data.isnull().sum()/len(data)*100

# # Item_Visibility: we make the zeroes to NaN
# data.loc[:, 'Item_Visibility'].mean()
# data.loc[:, 'Item_Visibility'] = data.loc[:, 'Item_Visibility'].replace(0.000000, np.nan)
# mean_item_vis = data.loc[:, 'Item_Visibility'].mean() # Now mean value has changed
# data.loc[:, 'Item_Visibility'].mean()

# # Item_Fat_content: we make the LF to Low Fat etc, so that we only have Low Fat and Regular
# data.loc[:'Item_Fat_Content'] = data.loc[:'Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat':'Low Fat', 'reg': 'Regular'})

# # And we drop the folowing columns
# # train.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year','Outlet_Location_Type'], axis=1, inplace=True)
# data.isnull().sum()/len(data)*100


# # Results
# data.head(40)
# data.info()

# for column in data.columns:
#     data[column].value_counts()


# + [markdown] hide_input=false
# ## 1.3 Ordinal data --> numeric data
# -

# If we have any ordinal data in our categorical group we have to transform it into numbers, and move it to the numerical group.

# __Ordinal data:__ Outlet_Size <br>
# __Nominal data:__ Item_Fat_Content, Item_Type, Outlet_Location_Type, Outlet_Type

# +
# # Ordinal data - Outlet_Size, ref.PML p.113
# data['Outlet_Size']

# size_mapping = {
#                 'Small': 1,
#                 'Medium': 2,
#                 'High': 3
# }

# data['Outlet_Size'] = data.loc[:, ('Outlet_Size')].map(size_mapping)
# data['Outlet_Size']
# -

# <br>
# <hr>

# # 2. Visualizing data / Exploratory Data Analysis (EDA) 

# ## 2.1 Start by Naming variables - original column names or x1, x2.., y1, y2.? Choose

# Now before we display all our data we have the option to choose between 2 different namings of our variables. I.e. the long original name, or xc, xn, for categorial or numerical features, respectively.

# +
name_as_x = 'no' # Set this to 'yes' or 'no'

if name_as_x == 'yes':
    data.columns = col_xy
else:
    data.columns = col_orig
# data
col_xy
col_orig
# + [markdown] hide_input=false
# ## 2.2 Short naming - we make the option to pick a short name for some of the variables

# + hide_input=false
df_xc = data.iloc[:, input_classes['cx']]
df_xn = data.iloc[:, input_classes['nx']]
df_yn = data.iloc[:, input_classes['ny']]
df_c = data.iloc[:, input_classes['c']]
df_n = data.iloc[:, input_classes['n']]
df_x = data.iloc[:, input_classes['x']]
df_y = data.iloc[:, input_classes['y']]


# -

# ## 2.3 Generic Plots - plots which are generated automaticly

# ### 2.3.1 Visualising probability distribution of Numerical Variables

# + hide_input=false
def probDistNumVar():
    nrows = math.ceil(len(df_n.columns)/3)
    df_n.hist(bins=15, figsize=(15, nrows*4), layout=(nrows, 3))
    plt.show();

show_this_plot = 'on'
    
if plots == 'on' and show_this_plot == 'on':
    probDistNumVar()
# -

# ### 2.3.2 Visualising Categorical Variables

# + hide_input=false
# Choose df_c or df_xc

from MLDA.plot_functions.functions import plotHistCatVar

show_this_plot = 'on'
    
if plots == 'on' and show_this_plot == 'on':
    plotHistCatVar(df_c)


# -

# ### 2.3.3 Visualising Relationships Between Numerical Variables

# + hide_input=false
def pairPlotWithHue():
    """This plot is used for plotting numerical data relationships - if categoric vars the hue option is used
    Understanding the plot below: the plot below for one single var: sns.pairplot(data, hue='Cat_var')
    so we iterate through all the categorical columns names and use each of the categories as hue's"""
    if len(df_c.columns) > 0:
        for var in df_c.columns[:len(df_c.columns)]:
            sns.pairplot(data, hue=var)
    else:
        sns.pairplot(data)
    plt.show();
        
show_this_plot = 'on'
    
if plots == 'on' and show_this_plot == 'on':
   pairPlotWithHue()
# -

# ### 2.3.4 Visualising Relationships Between Numerical and Categorical Variables 

# + hide_input=false
# data.iloc[:, input_classes['y']].columns
# data.iloc[:,input_classes['cx']].columns
data.iloc[:,input_classes['c']].columns
df_c.columns

# + hide_input=false
# Boxplot of each of the categorial variables and the numeric variables
# Categorial classes ordered by numeric variable

from MLDA.plot_functions.functions import boxplotCategoric_vsNumeric

# Function: boxplotCategoric_vsNumeric(data, col_c, col_n)

# Choose either all categorical var vs all numerical var, 
# Or choose only X cat and Y num
# col_c=df_c.columns, col_n=df_n.columns

show_this_plot = 'on'
    
if plots == 'on' and show_this_plot == 'on':
    boxplotCategoric_vsNumeric(data, df_c.columns, df_n.columns)


# -

# ## 2.4 Non-generic plots - plots where you have to choose variable parameters

# +
# # conditional relationships - the data is spread out
# col_orig
# col_xy

# Plot explanation: besides dataframe and col_wrap, this plot takes 4 inputs: 2 categorical and 2 numerical. Below
# is shown where to put the variables
# def plotChooseVars():
#     cond_plot = sns.FacetGrid(data=data, col='cat1_var', hue='cat2_var', col_wrap=4)
#     cond_plot.map(sns.scatterplot, 'num1_var', 'num2_var');

def plotChooseVars():
    cond_plot = sns.FacetGrid(data=data, col='Item_Type', hue='Outlet_Size', col_wrap=4)
    cond_plot.map(sns.scatterplot, 'Item_MRP', 'Item_Outlet_Sales');
    plt.legend(loc='upper right')
    plt.show()

show_this_plot = 'off'
    
if plots == 'on' and show_this_plot == 'on':
    plotChooseVars()
# -

# <br>
# <hr>

# # 3. Statistics

x # features
y # dependent var
data.iloc[:, x] # we see we have text values in our categorical data
df_x.iloc[2]

# +
# check our data for unique values etc

# Check the percentage of missing values for each parameter
data.isnull().sum()/len(data)*100 # we have 3 columns with missing values 

# Based on the shown data we have 3 columns with some NaN values

data.describe()
data.info()

for column in data.columns:
    data[column].value_counts()


# -

# ## 3.1 Heatmapping correlation and p-values

# ### 3.1.1 First we call the correlation function from corr_stats

# +
def calcCorrelationsForHeatmaps():
    from MLDA.corr_stats.corr_heatmap import correlation
    c = df_c.columns.to_list() # categorical
    n = df_n.columns.to_list() # numerical
    corr = correlation(data, catcols=c, numcols=n, CI=.1, method_cc='Asym', method_nn='Spearmann')
    return corr
show_this_plot = 'on'
    
if plots == 'on' and show_this_plot == 'on':
    corr = calcCorrelationsForHeatmaps()
    corr[0] # correlation values
    corr[1] # p values


# -

# ### 3.1.2 Heatmapping the correlation values

# +
def plotCorrelationValues():
    correlation = corr[0].loc[:, :].replace('p > CI', 0)
    correlation = correlation.loc[:, :].fillna(0)

    cmap = sns.diverging_palette(275, 150,s=80, l=55, n=9)
    g = sns.heatmap(correlation, vmax=.6, center=0, square=True,
                   linewidths=.5, cbar_kws={'shrink': .5}, annot=True,
                   fmt='.2f', cmap=cmap)
    # sns.despine()
    g.figure.set_size_inches(14, 10)

    # Due to bug in matplotlib 3.1.1
    bottom, top = g.get_ylim()
    g.set_ylim(bottom + 0.5, top - 0.5)

    # place xticks at the top
    g.xaxis.set_ticks_position('top')
    # rotate the top text
    g.set_xticklabels(g.get_yticklabels(), rotation = 60);
    plt.show();

show_this_plot = 'on'
    
if plots == 'on' and show_this_plot == 'on':
    plotCorrelationValues()


# -

# ### 3.1.3 Heatmapping the p values

# +
def plotPvalues():
    correlation = corr[1].loc[:, :].fillna(1)

    g = sns.heatmap(correlation, vmax=.2, center=0.05, square=True,
                   linewidths=.5, cbar_kws={'shrink': .5}, annot=True,
                   fmt='.2f', cmap='Greens_r')
    # sns.despine()
    g.figure.set_size_inches(14, 10)

    # Due to bug in matplotlib 3.1.1
    bottom, top = g.get_ylim()
    g.set_ylim(bottom + 0.5, top - 0.5)

    # place xticks at the top
    g.xaxis.set_ticks_position('top')
    # rotate the top text
    g.set_xticklabels(g.get_yticklabels(), rotation = 60);
    plt.show();

show_this_plot = 'on'
    
if plots == 'on' and show_this_plot == 'on':
    plotPvalues()
# -

# # Training our model with the 'data' dataframe

# +
from MLDA.miscellaneous.save_and_load_files import load_object, save_object

save_object(object_to_save=data, filename='energy_data')

e_data = load_object('energy_data.sav')
e_data

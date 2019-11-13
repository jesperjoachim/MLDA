# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os

try:
    os.chdir(os.path.join(os.getcwd(), "../../../../tmp"))
    print(os.getcwd())
except:
    pass
# %%
from MLDA.imports.DA_modules import *

# %% [markdown]
# # 0. Loading Data and First Look

# %%
# read the data
data_input = pd.read_csv("~/Work/MLDA_app/MLDA/input_data/Train_UWu5bXk.csv")
# train.head(40)

# column names
for column in data_input.columns:
    print(column)

# If no 'column names' or 'wrong names' in data_input, changes these

# %% [markdown]
# ## 0.1 Choosing the relevant variables

# %%
data_input.head()  # Columns to choose from
choose_var = [1, 2, 3, 4, 5, 8, 10, 11]  # First column: 0
data = data_input.iloc[:, choose_var]
data.head(25)

# %% [markdown]
# # 1. Data Munging
# %% [markdown]
# ## 1.1 Getting an Overview
# __Conclusion based on cell below:__ <br>
# 1) write... <br>
# 2) write... <br>

# %%
# Check the percentage of missing values (NaN values) for each parameter
data.isnull().sum() / len(data) * 100
# Conclusion: Item_weight and Outlet_size has missing values: 17% and 28 %, respevtively

data.describe()
data.info()

for column in data.columns:
    data[column].value_counts()

# %% [markdown]
# ### 1.1.1 Munging the 2 columns: Item_Fat_Content and Item_Visibility

# %%
data.isnull().sum() / len(data) * 100

# Item_Visibility: we make the zeroes to NaN
data.loc[:, "Item_Visibility"].mean()
data.loc[:, "Item_Visibility"] = data.loc[:, "Item_Visibility"].replace(
    0.000000, np.nan
)
mean_item_vis = data.loc[:, "Item_Visibility"].mean()  # Now mean value has changed
data.loc[:, "Item_Visibility"].mean()

# Item_Fat_content: we make the LF to Low Fat etc, so that we only have Low Fat and Regular
data.loc[:"Item_Fat_Content"] = data.loc[:"Item_Fat_Content"].replace(
    {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
)

# And we drop the folowing columns
# train.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year','Outlet_Location_Type'], axis=1, inplace=True)
data.isnull().sum() / len(data) * 100


# Results
data.head(40)
data.info()

for column in data.columns:
    data[column].value_counts()

# %% [markdown]
# ### Separating data in categorial/numeric classes and x/y classes, and combination of these

# %%
# Pick column numbers from data, NOT from data_input, first column is 0 (zero)
categoric_x = [1, 3, 6]
numeric_x = [0, 2, 4, 5]
categoric_y = []
numeric_y = [7]

################ Calc below ###############
input_classes = {}  # init a dict
x = categoric_x + numeric_x
y = categoric_y + numeric_y
categoric = categoric_x + categoric_y
numeric = numeric_x + numeric_y

# Adding to the dict
input_classes["cx"] = categoric_x
input_classes["nx"] = numeric_x
input_classes["x"] = x
input_classes["cy"] = categoric_y
input_classes["ny"] = numeric_y
input_classes["y"] = y
input_classes["c"] = categoric
input_classes["n"] = numeric
x

# %% [markdown]
# ### Ordinal data --> numeric data
# %% [markdown]
# If we have any ordinal data in our categorical group we have to transform it into numbers, and move it to the numerical group.
# %% [markdown]
# __Ordinal data:__ Outlet_Size <br>
# __Nominal data:__ Item_Fat_Content, Item_Type, Outlet_Location_Type, Outlet_Type

# %%
# Ordinal data - Outlet_Size, ref.PML p.113
data["Outlet_Size"]

size_mapping = {"Small": 1, "Medium": 2, "High": 3}

data["Outlet_Size"] = data.loc[:, ("Outlet_Size")].map(size_mapping)
data["Outlet_Size"]

# %% [markdown]
# ### Creating new names based on the variable's entity - i.e is it x, y and is it categorical or numerical

# %%
# Store the original column names and order
col_orig = data.columns

# Creating new names based on x, y and categoric, numeric
col_xc = [
    "xc" + str(num + 1) for num, item in enumerate(input_classes["cx"])
]  # the item part is not used
col_xn = ["xn" + str(num + 1) for num, item in enumerate(input_classes["nx"])]
col_yc = ["yc" + str(num + 1) for num, item in enumerate(input_classes["cy"])]
col_yn = ["yn" + str(num + 1) for num, item in enumerate(input_classes["ny"])]
col_xy = col_xc + col_xn + col_yc + col_yn
print("x+y:", x + y)
print("col_xy:", col_xy)

# Transferring the right (original) ordering of the variables to the new names (x, y)
order_orig = []

for order, item in enumerate(x + y):
    print(order, item)

# The Approach for the loop below:
# for each item in our new-name-list (col_xy) we find the original place/order of that item
# by looping through each number in the 'x+y'-list. The number in the 'x+y'-list is the
# original order of the items in the new-name-list (col_xy). By picking that number - for example
# say 0, which say is in the 5'th place, we know by looking in col_xy at 5'th place that this item
# was original the first one. We then append this item as the first one to the list: order_orig.
# Then we pick number two, which is 1, etc

# we loop through each number from 0 to total number of columns, len(x+y)
for num in range(len(x + y)):
    # So we find 0 first...
    for order, item in enumerate(x + y):
        # pick 0 in list: x+y: let's say it is in the 5th place
        if item == num:
            # then in col_xy in the 5th place we know that this item was orig the first one
            order_orig.append(col_xy[order])

# we restore the original order by replacing col_xy with order_orig
col_xy = order_orig

data

# %% [markdown]
# <br>
# <hr>
# %% [markdown]
# # 2. Visualizing data / Exploratory Data Analysis (EDA)
# %% [markdown]
# ## 2.1 Start by Naming variables - original column names or x1, x2.., y1, y2.? Choose
# %% [markdown]
# Now before we display all our data we have the option to choose between 2 different namings of our variables. I.e. the long original name, or xc, xn, for categorial or numerical features, respectively.

# %%
name_as_x = "no"  # Set this to 'yes' or 'no'

if name_as_x == "yes":
    data.columns = col_xy
else:
    data.columns = col_orig
data

# %% [markdown]
# ## 2.2 Short naming - we make the option to pick a short name for some of the variables

# %%
df_xc = data.iloc[:, input_classes["cx"]]
df_xn = data.iloc[:, input_classes["nx"]]
df_yn = data.iloc[:, input_classes["ny"]]
df_c = data.iloc[:, input_classes["c"]]
df_n = data.iloc[:, input_classes["n"]]
df_x = data.iloc[:, input_classes["x"]]
df_y = data.iloc[:, input_classes["y"]]

# %% [markdown]
# ## 2.3 Generic Plots - plots which are generated automaticly
# %% [markdown]
# ### 2.3.1 Visualising Numerical Variables

# %%
sns.set(
    style="whitegrid", palette="deep", font_scale=0.9, rc={"figure.figsize": [8, 5]}
)


# %%
fig_num_rows = math.ceil(len(df_n.columns) / 3)

df_n.hist(bins=15, figsize=(15, 6), layout=(fig_num_rows, 3))

# %% [markdown]
# ### 2.3.2 Visualising Categorical Variables

# %%
fig_num_rows = math.ceil(len(df_xc.columns) / 4)

fig, ax = plt.subplots(fig_num_rows, 4, figsize=(20, 8))
for var, subplot in zip(df_c.columns, ax.flatten()):
    sns.countplot(df_c[var], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.show()

# %% [markdown]
# ### 2.3.3 Visualising Relationships Between Numerical Variables

# %%
# sns.pairplot([['Value','SprintSpeed','Potential','Wage']])
def make_plot():
    data.iloc[:, input_classes["cx"]].columns[:3]
    for var in data.iloc[:, input_classes["cx"]].columns[:3]:
        sns.pairplot(data, hue=var)
    plt.show()


# make_plot()

# %% [markdown]
# ### 2.3.4 Visualising Relationships Between Numerical and Categorical Variables

# %%
# data.iloc[:, input_classes['y']].columns
# data.iloc[:,input_classes['cx']].columns
col_yn
df_yn.columns


# %%
# We make boxplot of each of the categorial input variables and the numeric output variables
# Categorial classes ordered by increasing outlet price

# Number of rows
def plotXc_vs_Yn():
    fig_num_rows = math.ceil(len(df_xc.columns) / 3) * len(df_yn.columns)

    fig, ax = plt.subplots(fig_num_rows, 3, figsize=(15, 10))
    for output_var in df_yn.columns:
        for var, subplot in zip(df_xc.columns, ax.flatten()):
            sorted_ = data.groupby([var])[output_var].median().sort_values()
            sns.boxplot(
                x=var, y=output_var, data=data, ax=subplot, order=list(sorted_.index)
            )
            # sns.despine(trim=True, offset=2)
            for label in subplot.get_xticklabels():
                label.set_rotation(90)
    plt.tight_layout()
    plt.show()


# plotXc_vs_Yn()

# %% [markdown]
# ## 2.4 Non-generic plots - plots where you have to choose variable parameters

# %%
# conditional relationships
col_orig
col_xy


def plotChooseVars():
    cond_plot = sns.FacetGrid(data=data, col="xc2", hue="xn4", col_wrap=4)
    cond_plot.map(sns.scatterplot, "xn3", "yn1")
    plt.legend(loc="upper right")
    plt.show()


# plotChooseVars()

# %% [markdown]
# <br>
# <hr>
# %% [markdown]
# # 3. Statistics

# %%
x  # features
y  # dependent var
data.iloc[:, x]  # we see we have text values in our categorical data
df_x.iloc[2]


# %%
# check our data for unique values etc

# Check the percentage of missing values for each parameter
data.isnull().sum() / len(data) * 100  # we have 3 columns with missing values

# Based on the shown data we have 3 columns with some NaN values

data.describe()
data.info()

for column in data.columns:
    data[column].value_counts()


# %%
from MLDA.corr_stats.corr_heatmap import correlation

c = df_c.columns
n = df_n.columns
correlation(dataset=data, catcols=c, numcols=n)

# %% [markdown]
# # Training our Model
# %% [markdown]
# ### Separating X and y

# %%
# Now before we prepare the categorial data for further analysis we separate the predictor(s) (input) from the prediction (output)
# train_y = train["Item_Outlet_Sales"]
# train.drop(["Item_Outlet_Sales"], axis=1, inplace=True)
# train_X = train
# train_X.head()
# train_y.head()


# %%


# %% [markdown]
# ### Statistical Analysis

# %%


# %% [markdown]
# ### Dimension Reduction

# %%
# sns.pairplot(train_X)


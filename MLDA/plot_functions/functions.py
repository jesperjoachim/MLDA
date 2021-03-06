import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import math

"""Section for defining set values for plots and for accessing the values via functions"""

dict_boxplot_setvalues = {"label_size": 15, "labelpadding": 13}


def setSharedAxParameters(ax_instance):
    ax_instance.xaxis.label.set_size(dict_boxplot_setvalues["label_size"])
    ax_instance.xaxis.labelpad = dict_boxplot_setvalues["labelpadding"]
    ax_instance.yaxis.label.set_size(dict_boxplot_setvalues["label_size"])
    ax_instance.yaxis.labelpad = dict_boxplot_setvalues["labelpadding"]


def setSharedAx_xParameters(ax_instance):
    ax_instance.xaxis.label.set_size(dict_boxplot_setvalues["label_size"])
    ax_instance.xaxis.labelpad = dict_boxplot_setvalues["labelpadding"]


def setSharedAx_yParameters(ax_instance):
    ax_instance.yaxis.label.set_size(dict_boxplot_setvalues["label_size"])
    ax_instance.yaxis.labelpad = dict_boxplot_setvalues["labelpadding"]


"""END Section for defining set values for plots and for accessing the values via functions"""

"""Section with plot functions"""


def plotHistCatVar(df_cat_vars):
    """This function is used for plotting histograms of the categoric vars
    Input: df_cat_vars: pandas dataframe
    Output: plot """

    if len(df_cat_vars.columns) > 3:
        nrows = math.ceil(len(df_cat_vars.columns) / 3)
        ncols = 3
    else:
        nrows = 1
        ncols = len(df_cat_vars.columns)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 8))
    for var, ax in zip(df_cat_vars.columns, axs.flatten()):
        sns.countplot(df_cat_vars[var], ax=ax)
        plt.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.45
        )
        for label in ax.get_xticklabels():
            label.set_rotation(90)
        setSharedAxParameters(ax)
    plt.show()


def boxplotCategoric_vsNumeric(data, col_c, col_n):
    """This function is divided into 3 sections. First one is for only one cat variable, second is for 2 or 3, and the third is for cat variables above 3.
    Input: df: pandas dataframe, col_c/col_n: pandas Index
    Output: plot or string"""
    # For only one cat var, it is the number of numeric var that defines the number of cols in the fig.
    # If num var is 3 or above, ncols is equal to 3
    if len(col_c) == 1:
        nrows = math.ceil(len(col_n) / 3)
        # If num var is 3 or above, ncols is equal to 3
        ncols = [3 if len(col_n) >= 3 else len(col_n)][0]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
        cat = col_c[0]
        for ax, var in zip(axs.flatten(), col_n):
            sns.boxplot(x=cat, y=var, data=data, ax=ax)
            plt.subplots_adjust(
                left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.45
            )
            setSharedAxParameters(ax)
        plt.show()

    if len(col_c) <= 3 and len(col_c) > 1:
        # In this section we generate a single fig canvas onto which we plot all the axs

        # we generate the xy-pairs to be plotted.
        # The data structure mimics the axs data structure which is generated below
        xy_sets = []
        for n in col_n:
            row = []
            for c in col_c:
                row.append([c, n])
            xy_sets.append(row)

        # Creating the fig and axes objects onto which we add the xy-data
        fig, axs = plt.subplots(
            nrows=len(col_n),
            ncols=len(col_c),
            sharex=False,
            sharey="row",
            figsize=(10, 7),
        )
        # First we unpack the axs and xy_sets into each row
        for ax_rows, row in zip(axs, xy_sets):
            # Then we unpack the rows into their single ax-object and single xy-set, respectively
            for ax, xy_set in zip(ax_rows, row):
                # Then we match each single item in the boxplot
                sns.boxplot(x=xy_set[0], y=xy_set[1], data=data, ax=ax)
                # Remove all labels, then we add the right ones below
                ax.set_ylabel("")
                ax.set_xlabel("")
        # y-labels: in first column we want to show all the y-labels
        for ax, sets in zip(axs, xy_sets):
            ax[0].set_ylabel(sets[0][1])
            setSharedAx_yParameters(ax[0])
        # x-labels: in last row we want to show all the x-labels
        for ax, xy_set in zip(axs[-1], xy_sets[-1]):
            ax.set_xlabel(xy_set[0])
            setSharedAx_xParameters(ax)

        fig.align_labels()
        fig.tight_layout()
        plt.show()

    if len(col_c) > 3:
        # In this section we generate a fig with subplots for each of the numeric var
        cat_ax_per_num = math.ceil(len(col_c) / 3) * 3
        axs_total = cat_ax_per_num * len(col_n)
        nrows = math.ceil(len(col_c) / 3)
        for var in col_n:
            xy_sets = []
            for c in col_c:
                xy_sets.append([c, var])
            fig, axs = plt.subplots(nrows=nrows, ncols=3, sharey=True, figsize=(15, 10))
            if hasattr(axs, "flatten"):
                axs_flat = axs.flatten()
            for ax, xy_set in zip(axs_flat, xy_sets):
                sns.boxplot(x=xy_set[0], y=xy_set[1], data=data, ax=ax)
                # Remove y-labels, then we add the right ones below
                ax.set_ylabel("")
            # y-labels: in first column we want to show all the y-labels
            for ax, sets in zip(axs, xy_sets):
                ax[0].set_ylabel(sets[1])
            for ax in axs_flat:
                setSharedAxParameters(ax)
            plt.subplots_adjust(
                left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.45
            )
        plt.show()
    elif len(col_c) == 0:
        return "No categorical data"


"""END Section with plot functions"""


"""Testing"""

# data = pd.read_csv(
#     "https://raw.githubusercontent.com/CoreyMSchafer/code_snippets/master/Python/Matplotlib/10-Subplots/data.csv"
# )

# gender = [
#     "f",
#     "f",
#     "m",
#     "m",
#     "m",
#     "f",
#     "f",
#     "m",
#     "m",
#     "m",
#     "f",
#     "f",
#     "m",
#     "m",
#     "m",
#     "f",
#     "f",
#     "m",
#     "m",
#     "m",
#     "f",
#     "f",
#     "m",
#     "m",
#     "m",
#     "f",
#     "f",
#     "m",
#     "m",
#     "m",
#     "f",
#     "f",
#     "m",
#     "m",
#     "m",
#     "m",
#     "f",
#     "m",
# ]
# have_pet = [
#     "y",
#     "y",
#     "n",
#     "m",
#     "m",
#     "f",
#     "f",
#     "m",
#     "m",
#     "m",
#     "f",
#     "f",
#     "m",
#     "m",
#     "m",
#     "f",
#     "f",
#     "m",
#     "m",
#     "y",
#     "f",
#     "f",
#     "y",
#     "n",
#     "n",
#     "f",
#     "f",
#     "m",
#     "m",
#     "m",
#     "f",
#     "f",
#     "m",
#     "m",
#     "n",
#     "m",
#     "n",
#     "y",
# ]

# data["Gender"] = gender  # Adding categorical data
# data["Have_pet"] = have_pet  # Adding categorical data, again

# data["Gender2"] = gender  # Adding categorical data
# data["Have_pet2"] = have_pet  # Adding categorical data, again


# numeric = [0, 1, 2, 3]
# categoric = [4, 5, 6, 7]
# n = data.iloc[:, numeric]
# c = data.iloc[:, categoric]

# boxplotCategoric_vsNumeric(data, c.columns, n.columns)

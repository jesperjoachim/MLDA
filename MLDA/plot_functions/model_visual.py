import copy
import math
import numpy as np
import pandas as pd
from MLDA.miscellaneous.save_and_load_files import load_object, save_object

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.model_selection import LearningCurve


# def varPropDict(xdata_names=None, x=None, x_steps=None, y=None, y_steps=None):
#     """To keep track of the assigned properties we create a var_prop dict
#     Output example: {'X2': ['x', 10], 'X3': ['value', None], 'X4': ['y', 15], 'X5': ['value', None], 'X6': ['value', None],
#     'X7': ['value', None], 'X8': ['value', None]}
#     """
#     var_prop = {}
#     for name in xdata_names:
#         # if name == x means: if the name is equal to the str we assigned to x
#         if name == x_var:
#             var_prop[name] = ["x", x_steps]
#         elif name == y:
#             var_prop[name] = ["y", y_steps]
#         else:
#             var_prop[name] = ["value", None]  # typically 'value' is the opt_value
#     return var_prop


def makeLinspace(var_prop=None, dict_namesbounds=None):
    """Based on var_prop we can now create a names_and_linspace dict which creates a linspace for x and y
    Output example: {'X2': array([514.5, 661.5, 808.5]), 'X4': array([110.25  , 137.8125, 165.375 , 192.9375, 220.5   ])}
    """
    names_and_linspace = {}
    for key in var_prop:
        if var_prop[key][0] == "x":  # I.e if we have assigned x to this var, then:
            # add an entry to names_and_linspace with a linspace with steps as specified above
            names_and_linspace[key] = np.linspace(
                dict_namesbounds[key][0], dict_namesbounds[key][1], var_prop[key][1]
            )
        elif var_prop[key][0] == "y":
            names_and_linspace[key] = np.linspace(
                dict_namesbounds[key][0], dict_namesbounds[key][1], var_prop[key][1]
            )
    return names_and_linspace


def makePointsBasedOnDFxdata(df_xdata=None, var_str=None):
    # all unique train values of var_str, sorted and as array
    return df_xdata[var_str].drop_duplicates().sort_values().to_numpy()


def makePointsForXandY(df_xdata, xvar_str, yvar_str):

    """Function that makes a dict with names and points which can be passed to the meshgrid function.
    The array is made from the train data points.
    Input: 
    df_xdata: a dataframe with the train xdata
    xvar, yvar: strings with x and y variable names,
    Output:
    dict with names and points as np arrays
    """
    x_str, y_str = xvar_str, yvar_str  # to shorten names
    names_and_points = {}
    x_points, y_points = (
        makePointsBasedOnDFxdata(df_xdata=df_xdata, var_str=x_str),
        makePointsBasedOnDFxdata(df_xdata=df_xdata, var_str=y_str),
    )
    names_and_points[x_str] = x_points
    names_and_points[y_str] = y_points

    return names_and_points


# xy meshgrid
def meshgrid(xvar_str=None, yvar_str=None, names_and_points=None):
    x_points = names_and_points[xvar_str]  # x points to be plot on axis former linspace
    y_points = names_and_points[yvar_str]  # y points to be plot on axis former linspace
    X, Y = np.meshgrid(x_points, y_points)
    X_rows, X_cols = X.shape  # should ?correspond to (y, x) i Z
    # Note: X_rows = Y_rows and X_cols = Y_cols
    return X, Y, x_points, y_points


# Now making and collecting arrays


def makeAndCollectArrays(
    xvar_str=None,
    yvar_str=None,
    X=None,
    Y=None,
    x_points=None,
    y_points=None,
    xdata_names=None,
    dict_namesvalues=None,
):
    """Approach: for x and y assign X and Y from meshgrid, else assign an array with np.full()"""
    collect_arrays = {}
    for name in xdata_names:
        if name == xvar_str:
            collect_arrays[name] = X
        elif name == yvar_str:
            collect_arrays[name] = Y
        else:
            collect_arrays[name] = np.full(
                (len(y_points), len(x_points)), dict_namesvalues[name]
            )
    # Now ordering the arrays in respect to xdata_names
    array = [collect_arrays[key] for key in xdata_names]
    array = np.array(array)
    return array


# def makeAndCollectArrays(
#     var_prop=None,
#     X=None,
#     Y=None,
#     x_lin=None,
#     y_lin=None,
#     xdata_names=None,
#     dict_namesvalues=None,
# ):
#     """Approach: array for x and y assign X and Y from meshgrid, else assign an array with np.full()"""
#     collect_arrays = {}
#     for key in var_prop:
#         if var_prop[key][0] == "x":
#             collect_arrays[key] = X
#         elif var_prop[key][0] == "y":
#             collect_arrays[key] = Y
#         else:
#             collect_arrays[key] = np.full(
#                 (len(y_lin), len(x_lin)), dict_namesvalues[key]
#             )
#     # Now ordering the arrays in respect to xdata_names
#     array = [collect_arrays[key] for key in xdata_names]
#     array = np.array(array)
#     return array


def reshapeAndTranspose(array=None):
    """Reshaping array from (n_features, X_rows, X_cols) to (n_features, X_rows * X_cols)"""
    num_feat, X_rows, X_cols = array.shape
    array = array.reshape(num_feat, X_rows * X_cols)
    # # Transposing array from (n_features, X_rows * X_cols) to (X_rows * X_cols, n_features)
    array_T = np.transpose(array)
    return array_T


# def f(x):
#     return x[0] + x[1] + x[2] + x[3] + x[4] + x[5]


def makeZ(array=None, x_points=None, y_points=None, predictor=None):
    """Generate Z by passing arrays to predictor/function"""
    Z = predictor.predict((array))
    # Reshaping Z, so it's x part - i.e cols in Z.reshape(rows, cols) - macthes numbers of x_points
    # and rows macthes numbers of y_points
    Z = Z.reshape(len(y_points), len(x_points))
    return Z


def xyZ_forSurfaceplot(
    predictor=None,
    dict_namesbounds=None,
    dict_namesvalues=None,
    df_xdata=None,
    xvar_str=None,
    yvar_str=None,
    reshape_transpose=True,
):
    """
    Approach: in this function we collect many data and calc Z

    Steps: 1) Use dict to assign x, y to vars; 2) make linspace for x, y; 3a) meshgrid; 3b) make and collect arrays for all feat/vars;
    4) reshape and transpose array; 5) generate Z
    Inputs: 
    predictor: choose function, eg. f;  
    
    """
    xdata_names = df_xdata.columns  # [feature2, feature3, feature4, ....feature N]
    # Assign the property to each of the variables

    # step 1
    vars_points_dict = makePointsForXandY(
        df_xdata=df_xdata, xvar_str=xvar_str, yvar_str=yvar_str
    )
    # step 2
    X, Y, x_points, y_points = meshgrid(
        xvar_str=xvar_str, yvar_str=yvar_str, names_and_points=vars_points_dict
    )
    # step 3
    array = makeAndCollectArrays(
        xvar_str=xvar_str,
        yvar_str=yvar_str,
        X=X,
        Y=Y,
        x_points=x_points,
        y_points=y_points,
        xdata_names=xdata_names,
        dict_namesvalues=dict_namesvalues,
    )
    # step 4
    array = reshapeAndTranspose(array)
    # step 5
    Z = makeZ(array=array, x_points=x_points, y_points=y_points, predictor=predictor)
    return x_points, y_points, Z


def surfacePlotly(
    x_points=None,
    y_points=None,
    Z_array=None,
    xvar_str=None,
    yvar_str=None,
    x_opt=None,
    y_opt=None,
    z_opt=None,
):
    fig = go.Figure(
        data=[
            go.Surface(z=Z_array, x=x_points, y=y_points),
            go.Scatter3d(
                x=x_opt,
                y=y_opt,
                z=z_opt,
                mode="markers+text",
                name="Markers and Text",
                text=["Text D"],
                textposition="bottom center",
            ),
        ]
    )
    # Layout
    fig.layout = dict(
        font=dict(family="Courier New, monospace", size=11, color="#7f7f7f"),
        autosize=False,
        width=600,
        height=600,
        scene=dict(
            zaxis=dict(
                range=[np.min(Z_array) * 0.9, np.max(Z_array) * 1.1], autorange=False
            ),
            xaxis=dict(title=f"{xvar_str}"),
            yaxis=dict(title=f"{yvar_str}"),
            aspectratio=dict(x=1, y=1, z=1),
        ),
    )
    fig.show()


# print(x_points, y_points, Z)
# surfacePlotly(
#     x_points=x_points,
#     y_points=y_points,
#     Z_array=Z,
#     xvar_str="height",
#     yvar_str="roof_area",
#     x_opt=(3.5,),
#     y_opt=(220.5,),
#     z_opt=(6.048,),
# )

"""Creating animation plot"""


def forAnimationPlot(
    predictor=None,
    dict_namesbounds=None,
    dict_namesvalues=None,
    df_xdata=None,
    xvar_str=None,
    yvar_str=None,
    frame_var_str=None,
    reshape_transpose=True,
):
    """Approach: we keep changing the value in dict_namesvalues for the frame_var and calc Z for each"""
    Z_frames_dict = {}
    var = frame_var_str
    var_lowbound, var_highbound = dict_namesbounds[var]
    # Make points for the frame_var
    num_frames = makePointsBasedOnDFxdata(df_xdata=df_xdata, var_str=var)
    # Copying dict_namesvalues in order to leave dict_namesvalues unchanged
    dict_namesvalues_copy = copy.deepcopy(dict_namesvalues)
    for frame in num_frames:
        # Here we change dict_namesvalues and replace the frame_var with each value in num_frames
        dict_namesvalues_copy[var] = frame
        # With dict_namesvalues being modified we calc Z
        x_points, y_points, Z = xyZ_forSurfaceplot(
            predictor=predictor,
            dict_namesbounds=dict_namesbounds,
            dict_namesvalues=dict_namesvalues_copy,
            df_xdata=df_xdata,
            xvar_str=xvar_str,
            yvar_str=yvar_str,
            reshape_transpose=reshape_transpose,
        )
        # Collecting Z for each frame (i.e each value in num_frames)
        Z_frames_dict[frame] = Z
    return x_points, y_points, num_frames, Z_frames_dict


def plotlyAnimation(
    xvar_str=None,
    yvar_str=None,
    frame_var_str=None,
    x_points=None,
    y_points=None,
    frame_points=None,
    Z_dict=None,
):
    """"""
    Z = [Z_dict[key] for key in Z_dict]
    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(
                    # x=x_lin, y=y_lin,
                    z=Z_dict[key]
                ),  # update only z; x, y are the same as in basic data
                #     traces= [0],
                #     cmax=zmax,
                #     cmin=zmin,
                name=f"{key:.0f}",
            )
            for key in Z_dict
        ]
    )

    # Add data to be displayed before animation starts
    fig.add_trace(
        go.Surface(
            x=x_points,
            y=y_points,
            z=[Z_dict[key] for key in Z_dict][0],
            cmax=np.max(Z) * 1.1,
            cmin=np.min(Z) * 0.9,
        )
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "currentvalue": {"prefix": f"{frame_var_str}:"},
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": f.name,
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.layout = dict(
        font=dict(family="Courier New, monospace", size=11, color="#7f7f7f"),
        autosize=False,
        width=600,
        height=600,
        scene=dict(
            zaxis=dict(range=[np.min(Z) * 0.9, np.max(Z) * 1.1], autorange=False),
            xaxis=dict(title=f"{xvar_str}"),
            yaxis=dict(title=f"{yvar_str}"),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(850)],  # frame duration
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    fig.update_layout(
        width=900,
        margin=dict(r=1, l=1, b=1, t=1),
        title="Plot Title",
        xaxis_title="x Axis Title",
        yaxis_title="y Axis Title",
        showlegend=True,
    )
    fig.show()


# print(x_points, y_points, num_frames, Z_frames_dict)


# plotlyAnimation(
#     xvar_str="height",
#     yvar_str="roof_area",
#     frame_var_str="wall_area",
#     x_points=x_points,
#     y_points=y_points,
#     frame_points=num_frames,
#     Z_dict=Z_frames_dict,
# )

"""END Creating animation plot"""


"""Validating the generated data"""


def calc_n_percentOfMax(n_percent=None, bounds=[]):
    return bounds[0] + n_percent * (bounds[1] - bounds[0])


def generateLevelsDict(
    levels=None,
    vars=None,
    level_vars=None,
    df_xdata=None,
    dict_namesbounds=None,
    dict_namesvalues=None,
):
    """
    Output example: {'0%': [674.5, 245.0, 110.25, 3.5, 4.0, 0.0, 0.0], '100%': [674.5, 245.0, 220.5, 7.0, 4.0, 0.0, 0.0]}
    """
    collect_values = {}
    for entry in levels:
        levels_dict = copy.deepcopy(dict_namesvalues)
        for var in level_vars:
            bounds = dict_namesbounds[var]
            n_percent_of_max = calc_n_percentOfMax(bounds=bounds, n_percent=entry)
            levels_dict[var] = n_percent_of_max
        values_ordered = []
        for feat in df_xdata.columns.to_list():
            values_ordered.append(levels_dict[feat])
        collect_values[str(entry * 100) + "%"] = values_ordered
    return collect_values


def generate_zValues(
    predictor=None,
    feat_to_vary=None,
    x_linspace=None,
    all_feat_xvalues=None,
    df_xdata=None,
):
    # First we find the index position of the feature to vary
    pos_var = df_xdata.columns.to_list().index(feat_to_vary)
    collection_of_xvalues = []  # list to be filled with sets for each x in x_linspace
    for entry in x_linspace:
        x_values = all_feat_xvalues[:]  # since lists are mutable we make a copy
        x_values[pos_var] = entry
        collection_of_xvalues.append(x_values)
    x = np.array(collection_of_xvalues)
    z = predictor.predict(x)
    return z


# test_genzval = generate_zValues(predictor=f, feat_to_vary='X2', x_linspace=[515, 600, 808], all_feat_xvalues=None, df_xdata_names=None)


def collect_xzValues(
    vars=None,
    var=None,
    x_linspace=None,
    levels=None,
    level_vars=None,
    df_xdata=None,
    dict_namesbounds=None,
    dict_namesvalues=None,
    predictor=None,
    all_feat=None,
):
    """Function which collects xz-values for specified variable (var). The xz_values for var
    are calc and collected for different sets of values of the variables specified in vars. The number of
    sets calc and collected is specified in the 'levels' parameter"""
    opt_values = []
    for feat in df_xdata.columns:
        opt_values.append(dict_namesvalues[feat])
    collect_xz_values = {"x": x_linspace, "z": {}}
    levels_dict = generateLevelsDict(
        levels=levels,
        level_vars=level_vars,
        vars=vars,
        df_xdata=df_xdata,
        dict_namesbounds=dict_namesbounds,
        dict_namesvalues=dict_namesvalues,
    )
    # Besides the levels we added by using generateLevelsDict we add the "opt_val" entry to levels_dict
    levels_dict["opt_val"] = opt_values

    for key in levels_dict:
        z_values = generate_zValues(
            predictor=predictor,
            feat_to_vary=var,
            x_linspace=x_linspace,
            all_feat_xvalues=levels_dict[key],
            df_xdata=df_xdata,
        )
        collect_xz_values["z"][key] = z_values

    return collect_xz_values


# def generateNamesAndArrays(vars=None, steps=None, dict_namesbounds=None):
#     """Creates a dict with names and the corresponding arrays for variables specified in vars.
#     Vars is normally equal to x,y,frame_var. Steps are the corresponding steps for the choosen vars"""
#     if len(vars) != len(steps):
#         return "Error: len(vars)!=len(steps)"
#     names_and_points = dict()
#     # Loop through each vars
#     for var, step in zip(vars, steps):
#         array = np.linspace(dict_namesbounds[var][0], dict_namesbounds[var][1], step)
#         names_and_points[var] = array
#     return names_and_points


def generateNamesAndPoints(vars=None, df_xdata=None):
    """Creates a dict with names and the corresponding points for variables specified in vars.
    Vars is normally equal to x,y,frame_var."""
    names_and_points = dict()
    # Loop through each vars
    for var in vars:
        points = makePointsBasedOnDFxdata(df_xdata=df_xdata, var_str=var)
        names_and_points[var] = points
    return names_and_points


def dataToPlot(
    vars=None,
    df_xdata=None,
    dict_namesvalues=None,
    dict_namesbounds=None,
    levels=None,
    level_vars=None,
    predictor=None,
):
    """This function have 2 parts: first part collects xy-data for the vars, second is converting it to a DF
    Output: data_to_plot: with x,y,frame_var='X4','X5','X2' and : x_steps, y_steps, num_frames = 10,15,20 and levels=[0, 1]:
    {'X4': {'x': array([110.25 , 165.375, 220.5  ]), 'z': {'0%': array([10.567638, 10.899158,  6.033828]), '100%': array([17.225954, 17.999774, 13.134444]), 'opt_val': array([10.567638, 10.899158,  6.033828])}}, 'X5': {'x': array([3.5       , 4.66666667, 5.83333333, 7.        ]), 'z': {'0%': array([10.567638, 10.567638, 17.225954, 17.225954]), '100%': array([ 6.033828,  6.033828, 13.134444, 13.134444]), 'opt_val': array([ 6.033828,  6.033828, 13.134444, 13.134444])}}, 'X2': {'x': array([514.5, 588. , 661.5, 735. , 808.5]), 'z': {'0%': array([12.014344, 13.065804, 15.233534, 10.88558 , 12.113366]), '100%': array([14.5351  , 15.52194 , 18.09691 , 13.408816, 14.380368]), 'opt_val': array([ 9.687934, 10.084234, 10.996294,  6.48174 ,  8.227986])}}}
    """
    data_to_plot = {}  # Dict to collect xz-data for the vars
    names_and_points = generateNamesAndPoints(vars=vars, df_xdata=df_xdata)
    # We call the collect_xzValues() function for all vars
    for var in vars:
        data_to_plot[var] = collect_xzValues(
            vars=vars,
            var=var,
            x_linspace=names_and_points[var],
            levels=levels,
            level_vars=level_vars,
            df_xdata=df_xdata,
            dict_namesvalues=dict_namesvalues,
            dict_namesbounds=dict_namesbounds,
            predictor=predictor,
        )
    # Converting to DF
    df_dict = {}
    for var in data_to_plot:
        df_dict[var] = pd.DataFrame({"x": data_to_plot[var]["x"]})
        for key in data_to_plot[var]["z"]:
            df_dict[var][key] = data_to_plot[var]["z"][key]
    return data_to_plot, df_dict


def removeVar(var=None, level_vars=None):
    labels = level_vars[:]
    if var in labels:
        labels.remove(var)
    return labels


def validationPlot(
    vars=None,
    df_xdata=None,
    dict_namesvalues=None,
    dict_namesbounds=None,
    levels=None,
    level_vars=None,
    predictor=None,
):
    data_to_plot, df_dict = dataToPlot(
        vars=vars,
        df_xdata=df_xdata,
        dict_namesvalues=dict_namesvalues,
        dict_namesbounds=dict_namesbounds,
        levels=levels,
        level_vars=level_vars,
        predictor=predictor,
    )
    fig, axs = plt.subplots(
        nrows=1, ncols=3, figsize=(15, 10), sharex=False, sharey="row"
    )
    for ax, var in zip(axs, vars):
        labels = removeVar(var=var, level_vars=level_vars)
        x = data_to_plot[var]["x"]
        for key in data_to_plot[var]["z"]:
            z = df_dict[var][key]
            sns.lineplot(x=x, y=z, data=df_dict[var], ax=ax, label=f"{labels}, {key}")
            ax.set_title(f"z vs {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("z")
            ax.legend(fontsize=8)
    plt.show()


"""END Validating the generated data"""


"""Find Variables For Plot"""


def findHighestFeatImportance(df_feat_imp=None, var_candidates=None):
    """Function that returns the highest value from the feature importance df when only the 
    var_candidates are left to choose from.
    Input: 
    df_feat_imp: single column df with feature importance of the vars
    var_candidates: list with var_candidates to choose from
    Output: string with index name of highest value.
    """
    # Find the column name of the feature importance df
    col_name = df_feat_imp.columns.to_list()[0]
    # Selecting only the rows in df_feat_imp that are in var_candidates
    df_feat_imp = df_feat_imp.loc[var_candidates]
    # Using idxmax() to find the max value in the column with name 'col_name' and return the name of that column
    return df_feat_imp.loc[df_feat_imp[col_name].idxmax()].name


def findCorrBelowLevelBetweenVars(
    df_corr=None, var_candidates=None, vars_chosen=None, level=None
):
    """Function that returns list with correlation.
    Input: 
    df_corr: symmetrical df with intercorrelation values between vars
    var_candidates: list with var_candidates to choose from
    vars_chosen: dict with vars already chosen
    level: correlation level to be below in order to pass
    Output: list with vars that passes the level test.
    """
    # Convert vars_chosen from dict to list
    vars_chosen_list = [vars_chosen[var] for var in vars_chosen]
    # Make df from df_corr but with rows and columns selected by var_candidates and vars_chosen_list, respectively.
    # That's because we are only interested in the correlation value to the other vars for the vars we have already chosen
    df = df_corr.loc[var_candidates, vars_chosen_list]
    # If we have a cell with "p > CI", i.e. low corr we replace this with a zero
    df.replace("p > CI", 0, inplace=True)
    # Then we drop every cell that is below the specified correlation level, level
    df = df[(df[vars_chosen_list] < level) & (df[vars_chosen_list] > -1 * level)]
    # Suppose we have 2 vars in vars_chosen. Then if var1 in df_corr passes the level for correlation
    # but the other var does not, we have now nan value in the cell that did not pass the level test,
    # so we drop that row by using dropna()
    df = df[vars_chosen_list].dropna()
    if df.empty:
        return f"var_candidates = {var_candidates}, vars_chosen_list = {vars_chosen_list} has no candidates that passes a level of {level}"
    else:
        return df.index.to_list()


def screenForCorrLevelAndFeatImp(
    df_feat_imp=None, df_corr=None, var_candidates=None, vars_chosen=None, level=None
):
    """Combines the functions findCorrBelowLevelBetweenVars and findHighestFeatImportance. I.e. first 
    screening for correlation below level, second find the highest feat imp of the list passing the screening.
    Input/Output: as for the two functions above.
    """
    # Screening: list with vars that has correlation below level
    below_level = findCorrBelowLevelBetweenVars(
        df_corr=df_corr,
        var_candidates=var_candidates,
        vars_chosen=vars_chosen,
        level=level,
    )
    # Check if below_level is empty - i.e. findCorrBelowLevelBetweenVars returns a string
    if isinstance(below_level, str):
        return below_level
    # Find highest amongst below_level list
    most_important = findHighestFeatImportance(
        df_feat_imp=df_feat_imp, var_candidates=below_level
    )
    return most_important


def findVarsForPlot(
    num_for_plot=None, xdata_names=None, df_corr=None, df_feat_imp=None, level=0.5
):
    """Function that returns dict with most relevant vars to plot according to the following princip:
    First pick the var with the highest feature importance, second: for this var find the correlation
    level to the other vars, and pick the vars that are above the specified correlation level, third:
    # from the list of vars above the specified correlation level choose the one with the highest feature 
    importance. Now two vars has been picked. If a third one is needed (i.e. num_for_plot=3), it goes 
    through second and step again but know with two vars to check for correlation to the other vars.
    Input: 
    num_for_plot: integer: 2 or 3
    xdata_names: name of the vars to screen between, list or pandas Index
    df_corr: symmetrical df with intercorrelation values between vars
    df_feat_imp: single column df with feature importance of the vars
    level: correlation level to be below in order to pass
    Output:  dict with most relevant vars to plot.
    """
    var_candidates = copy.deepcopy(xdata_names)
    if isinstance(var_candidates, pd.Index):
        var_candidates = var_candidates.to_list()
    vars_chosen = {}
    # Find first variable
    most_important = findHighestFeatImportance(
        df_feat_imp=df_feat_imp, var_candidates=var_candidates
    )
    var_candidates.remove(most_important)
    vars_chosen["var1"] = most_important
    # Find second variable or second and third depending on num_for_plot
    for num in range(num_for_plot - 1):
        most_important = screenForCorrLevelAndFeatImp(
            df_feat_imp=df_feat_imp,
            df_corr=df_corr,
            var_candidates=var_candidates,
            vars_chosen=vars_chosen,
            level=level,
        )
        # Check if most_important is a string - i.e. no candidates passes the test
        if "var_candidates =" in most_important:
            return most_important
        var_candidates.remove(most_important)
        vars_chosen[f"var{num+2}"] = most_important
    return vars_chosen


"""END Find Variables For Plot"""

"""Feature Importance"""


def showFeatureImportanceForModels(ML_dict=None):
    ydata_names = ML_dict["ydata_names"].to_list()
    nrows, ncols = math.ceil(len(ydata_names) / 2), len(ydata_names)
    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(14, 7), sharey=True)
    df_sorted = [
        ML_dict["Y"][y_name]["pred"]["forest"]["df_feat_imp"] for y_name in ydata_names
    ]
    for ydata_name, df, ax in zip(ydata_names, df_sorted, axs):
        #     print('df.columns', df.columns, len(df.columns))
        #     print('df.index', df.index, len(df.index))
        #     print('df', df)
        sns.barplot(y=df.columns[0], x=df.index, data=df, ax=ax)
        ax.tick_params(axis="both", colors="b", labelsize=9)
        for item in ax.get_xticklabels():
            item.set_rotation(45)
    # fig, axs = plt.subplots(figsize=(10,7))

    # print(len(df_sorted[0].columns), df_sorted[0])
    # sns.barplot(y=df_sorted[0].columns, x=df_sorted[0].index, data=df_sorted[0], ax=axs)
    fig.suptitle("Feature Importance", fontsize=25, color="b")
    fig.show()


"""END Feature Importance"""
# ML_dictEN1 = load_object("/home/jesper/Work/MLDA_app/MLDA/jupyter_ML/ML_dictEN1.sav")


# showFeatureImportance(ML_dict=ML_dictEN1)

"""Residuals"""

# Instantiate the visualizer and use the ransac instance from the linear model
def showResidualsForModels(ML_dict=None):
    model_names = ML_dict["models"]
    ydata_names = ML_dict["ydata_names"].to_list()
    Xtrain = ML_dict["X"]["X_train"]
    Xtest = ML_dict["X"]["X_test"]
    ncols = len(model_names)
    for yname in ydata_names:
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(16, 12), sharey=True)
        for ax, model in zip(axs, model_names):
            predictor = ML_dict["Y"][yname]["pred"][model]["predictor"]
            vis_resid = ResidualsPlot(ax=ax, model=predictor)
            Ytrain = ML_dict["Y"][yname]["actual"]["train"]
            Ytest = ML_dict["Y"][yname]["actual"]["test"]
            vis_resid.fit(X=Xtrain, y=Ytrain)  # Fit the training data to the model
            vis_resid.score(X=Xtest, y=Ytest)
            ax.tick_params(axis="both", colors="b", labelsize=15)
            ax.set_ylabel(ylabel="Residuals", color="b")
        fig.suptitle(yname, fontsize=20, color="b")

        vis_resid.show()


"""END Residuals"""


"""Learning Curve"""


def showLearningCurveForModels(ML_dict=None):
    import warnings

    warnings.filterwarnings("ignore")
    Xtrain = ML_dict["X"]["X_train"]
    ynames = ML_dict["ydata_names"]
    models = ML_dict["models"]
    ncols = len(models)
    model_labels = [key for key in ML_dict["Y"]["heat_load"]["pred"]]
    for yname in ynames:
        fig, axs = plt.subplots(nrows=1, ncols=ncols, sharey=True, figsize=(16, 9))
        model_labels = [label for label in ML_dict["Y"][yname]["pred"]]
        for model, model_label, ax in zip(models, model_labels, axs):
            Ytrain = ML_dict["Y"][yname]["actual"]["train"]
            model = ML_dict["Y"][yname]["pred"][model]["pred_object"]
            vis_learn_curve = LearningCurve(model, scoring="r2", ax=ax)
            vis_learn_curve.fit(Xtrain, Ytrain)
            ax.tick_params(axis="both", colors="b", labelsize=15)
            ax.set_title(f"{model_label}", color="b")
            ax.set_ylabel(ylabel=f"{model_label}", color="b")
        fig.suptitle(yname, fontsize=20, color="b")
    vis_learn_curve.show()


"""END Learning Curve"""

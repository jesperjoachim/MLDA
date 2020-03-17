import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# from MLDA.miscellaneous.save_and_load_files import load_object, save_object


"""ML_dict structure"""
# ML_dict_Name1 = {'id': "xcols: [0, 1, 2, 3, 4, 5]_ycols: [0, 1]_['ransac', 'forest']_test_size: 0.2",
#                           'df_data': {dataframe with xdata and ydata}
#                           'xdata_names':
#                           'X': {'X_train': np.arrays
#                                 'X_test': np.arrays}
#                           'models': {'names': ['ransac', 'forest']} example
#                           'Y': {'yname1': {'pred':{'ransac': {'train': np.arrays, 'test': np.arrays, 'predictor': None, 'pred_object': RANSACRegressor(), 'MSE': value, 'R2': value}
#                                                    'forest': {'train': np.arrays, 'test': np.arrays, 'predictor': None, 'pred_object': RandomForestRegressor() ,'MSE': value, 'R2': value}}
#                                            'actual': {'train': np.arrays, 'test': np.arrays}}
#                                {'yname2': {'pred':{'ransac': {'train': np.arrays, 'test': np.arrays, 'predictor: None', 'MSE': value, 'R2': value}
#                                                    'forest': {'train': np.arrays, 'test': np.arrays, 'predictor: None', 'MSE': value, 'R2': value}}
#                                            'actual': {'train': np.arrays, 'test': np.arrays}}}}
#     }
"""END ML_dict structure"""

"""DA_dict structure"""

# da_dict structure
# da_dict = {'id_label': "CATCOLS: [] - NUMCOLS: ['rel_compact', 'surf_area', 'wall_area', 'roof_area', 'height', 'orientation', 'glazing_area', 'glaz_area_distrib', 'heat_load', 'cool_load'] - CI: 0.1 - METHOD_CC: Asym - METHOD_CN: Omega - METHOD_NN: Spearmann",
#            'data': df,
#            'correlation' df_corr,
#            'p-values': df_p}

"""END DA_dict structure"""


"""Adding stuff to ML_dict"""


def returnID_of_ML_dict(test_size=None, df_xdata=None, df_ydata=None, models=None):
    keyname_x = [num for num, name in enumerate(df_xdata.columns)]
    keyname_y = [num for num, name in enumerate(df_ydata.columns)]
    return f"xcols: {keyname_x}_ycols: {keyname_y}_{models}_test_size: {test_size}"


def splitDataAndAddToMLDict(
    df_xdata=None, df_ydata=None, ML_dict=None, models=None, test_size=None
):
    # df_ydata_names = [name for name in df_ydata.columns]
    df_data = pd.concat([df_xdata, df_ydata], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        df_xdata, df_ydata, test_size=test_size, random_state=1
    )
    entry_id = returnID_of_ML_dict(
        test_size=test_size, df_xdata=df_xdata, df_ydata=df_ydata, models=models
    )
    # First check if ID key exist at all - i.e. is there anything in ML_dict yet?
    if "ID" in ML_dict:
        # Then we check if the entry_id is in the 'ID' key - i.e. we have already chosen this data once
        if entry_id in ML_dict["ID"]:
            return f"!entry_id: {entry_id} already exists!"
    # If no ID key exist we add one
    else:
        ML_dict["ID"] = entry_id
    # Adding stuff to ML_dict
    # First adding the dataframe with both x and y data
    ML_dict["df_data"] = df_data

    # Next X, xdata_names and models
    ML_dict["xdata_names"] = df_xdata.columns
    ML_dict["X"] = dict(X_train=X_train.to_numpy(), X_test=X_test.to_numpy())
    ML_dict["ydata_names"] = df_ydata.columns
    ML_dict["models"] = models
    # Then Y
    ML_dict["Y"] = {}
    for name in df_ydata.columns:
        ML_dict["Y"][name] = dict(
            pred=dict(),
            actual=dict(train=Y_train[name].to_numpy(), test=Y_test[name].to_numpy()),
        )
    for name in ML_dict["Y"]:
        for model in models:
            ML_dict["Y"][name]["pred"][model] = dict(
                train=None, test=None, predictor=None, pred_object=None
            )
    return ML_dict, type(ML_dict)


def addFeatImpToDict(ML_dict=None):
    y_names = ML_dict["ydata_names"]
    for y_name in y_names:
        # Y_train_act = ML_dict["Y"][y_name]["actual"]["train"]
        # Y_test_act = ML_dict["Y"][y_name]["actual"]["test"]
        # for model in models:

        forest = ML_dict["Y"][y_name]["pred"]["forest"]["predictor"]
        importance = forest.feature_importances_
        xdata_names = ML_dict["xdata_names"].to_list()
        df_importance = pd.DataFrame(
            np.array(importance),
            index=xdata_names,
            columns=[f"{y_name}: Feature_Importance"],
        )
        sorted_ = df_importance.sort_values(by=f"{y_name}: Feature_Importance")
        ML_dict["Y"][y_name]["pred"]["forest"]["df_feat_imp"] = sorted_
    return ML_dict


"""END Adding stuff to ML_dict"""

# ML_dict = splitDataAndAddToMLDict(
#     df_xdata=xdata, df_ydata=ydata, ML_dict=ML_dict, models=models, test_size=test_size
# )
# print(ML_dict)
# # Save ML_dict

# ML_dict
# --------------------------------------------------------------------------------------------------

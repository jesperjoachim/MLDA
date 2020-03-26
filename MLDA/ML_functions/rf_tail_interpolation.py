import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from MLDA.miscellaneous.save_and_load_files import load_object, save_object

# Functions


def f(x=None, predictor=None):
    """Function that takes an array and a predictor and returns an prediction array. 
    Approach: if not already a numpy array then make it one. And if input is a single row
    then reshape input.
    Input: 
    x: numpy array, pandas df/series or python list
    predictor: sklearn predictor
    """
    if not hasattr(x, "reshape"):
        x = np.array(x)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    return predictor.predict(x)


def keepIfDataPointsAreComprisingPercentageAboveLimit(
    data_fraction_limit=None,
    list_of_values=None,
    percentage_of_range=None,
    xdata=None,
    var=None,
):
    """Function that takes list of values and returns another list of values which passed the conditions. The overall
    approach is to ensure that we base our interpolation on a representatative amount of data. In other words, if we
    want to make a prediction and one of our variables is not representet in our train dataset, i.e. not close to any
    of the train data that our model is based on, then we want to make a interpolation from data that our model is based on.
    And in order to say that these data are representatative is to say that they comprise a certain fraction of the total
    amount of data (data_fraction_limit), and that we collect the data for interpolation in a certain confined area (percentage_of_range). 
    Approach: the conditions to pass is the data_fraction_limit and percentage_of_range. The data_fraction_limit is
    the minimum fraction of data (of the total dataset) we want to base the points of interpolation on. The percentage_of_range
    is the distance range for the given variable, i.e. how far to the left and right we can go, when we collect data to calc 
    the data_fraction_limit. When data_fraction_limit is higher and the percentage_of_range is lower, the conditions are more difficult 
    to fullfill, since then we both want our data to represent a higher fraction of the total amount of data, while we also want to find these data on
    a small area (not far to the left and right). 
    Input: 
    data_fraction_limit:fraction we specify,
    list_of_values: values that are either rejected or returned by the function,
    percentage_of_range: fraction we specify,
    xdata: dataframe,
    var: variable
    Output: 
    list_result: values that passed the conditions.
    """
    list_result = []
    for item in list_of_values:
        low, high = item - percentage_of_range, item + percentage_of_range
        if (
            xdata[var][(xdata[var] > low) & (xdata[var] < high)].count()
            / xdata[var].count()
            > data_fraction_limit
        ):
            list_result.append(item)
    return list_result


def findBelowAndAboveValuesUsingRangeAndLimit(
    xdata=None,
    var=None,
    to_predict=None,
    percent_value_data_range=None,
    data_fraction_limit=None,
):
    list_lower = xdata[var][xdata[var] <= to_predict[var]].drop_duplicates()
    list_higher = xdata[var][xdata[var] >= to_predict[var]].drop_duplicates()
    # find data range for var
    lowest, highest = xdata[var].min(), xdata[var].max()
    l_h_range = highest - lowest  # lowest, highest range
    percentage_of_range = l_h_range * percent_value_data_range

    # llksadfl: list_lower_keep_since_above_data_fraction_limit
    llksadfl = keepIfDataPointsAreComprisingPercentageAboveLimit(
        data_fraction_limit=data_fraction_limit,
        list_of_values=list_lower,
        percentage_of_range=percentage_of_range,
        xdata=xdata,
        var=var,
    )
    # lhksadfl: list_higher_keep_since_above_data_fraction_limit
    lhksadfl = keepIfDataPointsAreComprisingPercentageAboveLimit(
        data_fraction_limit=0.05,
        list_of_values=list_higher,
        percentage_of_range=percentage_of_range,
        xdata=xdata,
        var=var,
    )
    return llksadfl, lhksadfl


def collectValuesForLinRegr(
    df_xdata=None,
    to_predict=None,
    threshold=0.1,
    percent_value_data_range=0.05,
    data_fraction_limit=0.02,
):
    """Approach: If prediction value (var) is not outside the threshold, keep the value (i.e. append it two times).
    When the prediction value (var) is not outside the threshold it means it is close to some of the models training data points.
    And this means again that no interpolation is needed. Else we will find interpolation points by finding points below and above
    the prediction value."""
    xdata = df_xdata
    to_predict_for_regr = {key: [] for key in xdata.columns}
    for var in to_predict:
        # First we find the closest lower/higher value to the prediction value (var),
        # NOTE: the closest value could be the prediction value.
        closest_lower = xdata[var][xdata[var] <= to_predict[var]].max()
        closest_higher = xdata[var][xdata[var] >= to_predict[var]].min()
        # Then we check if it inside the threshold - i.e. is either closest_lower or closest_higher sufficient close
        # to the prediction value. And keep the value if it is (i.e. append it two times).
        if closest_lower > to_predict[var] * (
            1 - threshold
        ) or closest_higher < to_predict[var] * (1 + threshold):
            for i in range(2):
                to_predict_for_regr[var].append(to_predict[var])
        else:
            # Arriving here in the algoritme means that no training data are
            # close to the prediction value for this var, and thus we will
            # find interpolation points by finding points below and above
            # the prediction value.
            # llksadfl: list_lower_keep_since_above_data_fraction_limit
            # lhksadfl: list_higher_keep_since_above_data_fraction_limit
            llksadfl, lhksadfl = findBelowAndAboveValuesUsingRangeAndLimit(
                xdata=xdata,
                var=var,
                to_predict=to_predict,
                percent_value_data_range=percent_value_data_range,
                data_fraction_limit=data_fraction_limit,
            )
            if len(llksadfl) == 0 or len(lhksadfl) == 0:
                return f"Unable to find interpolation data points given the input: percent_value_data_range={percent_value_data_range} and data_fraction_limit={data_fraction_limit}"
            else:
                # Now finding the value closest to the prediction value
                closest_lower = max(llksadfl)
                closest_higher = min(lhksadfl)
            # Now appending the data points below and above the prediction value
            to_predict_for_regr[var].append(closest_lower)
            to_predict_for_regr[var].append(closest_higher)
    return to_predict_for_regr


def makeDFforMultipleRegr(to_predict=None, df_xdata=None, threshold=0.1):
    """Function that makes a dataframe for multiple interpolation of vars.
    Approach: The function creates data for multiple linear regression by making 
    rows starting with 2 lower values for var. For the first low value of
    var we set all the other vars to their lower value in that row. For the
    next low value of var we set all the other vars to their higher value. 
    And repeating that for but with the 2 high values for var. And then the
    same for the next var.
    Input: 
    to_predict: dict with the values to predict Y,
    df_xdata: a dataframe with the train xdata
    threshold: threshold to pass to the function collectValuesForLinRegr
    """
    to_predict_for_regr = collectValuesForLinRegr(
        to_predict=to_predict, df_xdata=df_xdata, threshold=threshold
    )
    collect_to_df = {key: [] for key in to_predict_for_regr}
    count_vars_no_regr = 0
    for var in to_predict_for_regr:
        # If var in to_predict_for_regr have different values
        if to_predict_for_regr[var][0] != to_predict_for_regr[var][1]:
            # then we append these two values 2 times
            for i in range(2):
                collect_to_df[var].append(to_predict_for_regr[var][0])
            for i in range(2):
                collect_to_df[var].append(to_predict_for_regr[var][1])
            # then we fill out the other
            for name in to_predict_for_regr:
                if name == var:
                    continue
                for i in range(2):
                    collect_to_df[name].append(to_predict_for_regr[name][0])
                    collect_to_df[name].append(to_predict_for_regr[name][1])
        else:
            count_vars_no_regr += 1
    # if all vars are below the threshold - i.e. no interpolation
    if count_vars_no_regr == len(to_predict_for_regr):
        return "All vars are below the threshold - i.e. no interpolation"
    df_regr = pd.DataFrame(data=collect_to_df)
    df_regr = df_regr.drop_duplicates()
    return df_regr, to_predict_for_regr


def findSlopes(df_xdata_regr=None, predictor=None):
    """Find slopes for all the relevant vars. Using the adaptable function f."""
    # we copy df_xdata, since now we add Y to the dataframe
    df_regr = df_xdata_regr.copy()
    df_regr["Y"] = df_regr[df_xdata_regr.columns].apply(f, args=(predictor,), axis=1)
    X = df_regr[df_xdata_regr.columns].to_numpy()
    Y = df_regr[["Y"]].to_numpy()
    model = LinearRegression().fit(X, Y)
    return model.coef_, df_regr


def intepolate(df_regr=None, slope=None, to_predict=None, to_predict_for_regr=None):
    """Based on slopes and data, multiple var interpolation is made. 
    Approach: we start out with the Y-value when all X-values are lowest (interpol_Y1), then we add
    change in Y (delta_Y) due to change in X (interpol_X - interpol_X1).
    The function returns the prediction of Y.
    Input: 
    df_regr: dataframe with combinations of nearest lowest and nearest highest from to_predict_for_regr, incl. Y predictions.
    slope: array with slopes
    to_predict: dict with the values to predict Y
    to_predict_for_regr: dict with nearest lowest and nearest highest or - if predict value for var is below threshold - predict value twice.
    """
    # The first row in df is when all X-values are lowest, which we call interpol_Y1
    interpol_Y1 = df_regr.loc[0, ["Y"]].to_numpy()
    interpol_X1 = np.array([to_predict_for_regr[num][0] for num in to_predict_for_regr])
    # NOTE: To calc delta_Y we have to make sure that the vars in interpol_X1 and interpol_X
    # comes in the same order. To ensure that we iterate through 'to_predict_for_regr' in the
    # list comprehension for both interpol_X1 and interpol_X.
    interpol_X = np.array([to_predict[num] for num in to_predict_for_regr])
    delta_Y = np.sum(slope * (interpol_X - interpol_X1))
    return interpol_Y1 + delta_Y


def refineRF_PredictionByInterpolation(
    xdata_names=None,
    xdata_numpy=None,
    to_predict_dict=None,
    threshold=0.1,
    predictor=None,
):
    """Coordinator function. Function that predicts Y by calling the relevant functions.
    Input: 
    to_predict: dict with the values to predict Y,
    df_xdata: a dataframe with the train xdata
    threshold: threshold to pass to the function collectValuesForLinRegr
    """
    df_xdata = pd.DataFrame(data=xdata_numpy, columns=xdata_names)
    # First we check if all the vars are below the threshold
    if (
        makeDFforMultipleRegr(
            to_predict=to_predict_dict, df_xdata=df_xdata, threshold=threshold
        )
        == "All vars are below the threshold - i.e. no interpolation"
    ):
        return "All vars are below the threshold - i.e. no interpolation"

    df_xdata_regr, to_predict_for_regr = makeDFforMultipleRegr(
        to_predict=to_predict_dict, df_xdata=df_xdata, threshold=threshold
    )
    slope, df_regr = findSlopes(df_xdata_regr=df_xdata_regr, predictor=predictor)
    predict_Y = intepolate(
        df_regr=df_regr,
        slope=slope,
        to_predict=to_predict_dict,
        to_predict_for_regr=to_predict_for_regr,
    )
    return predict_Y[0][0]


"""Not Used"""


def findSlopeSingleVar(df_xdata_regr=None, predictor=None, var=None):
    """Find slope for when one var has two different values, all other vars have same values.
     Using the adaptable function f."""
    # we copy df_xdata, since now we add Y to the dataframe
    df_regr = df_xdata_regr.copy()
    # Calc Y by applying predictor to x array
    df_regr["Y"] = df_regr[df_xdata_regr.columns].apply(f, args=(predictor,), axis=1)
    # Since we only find the slope for a single value of the x values we
    # shrink the df to only include this x
    df_regr["X"] = df_regr[var]
    # Convert to numpy
    X = df_regr["X"].to_numpy().reshape(-1, 1)
    Y = df_regr[["Y"]].to_numpy()
    # Make regression
    model = LinearRegression().fit(X, Y)
    return model.coef_  # return slope of single xvar


# # Exercise
# dict_test = load_object(
#     "/home/jesper/Work/MLDA_app/MLDA/input_data/train_test_dict.sav"
# )
# xdata_names = dict_test["N1"]["xdata_names"]
# xdata_numpy = dict_test["N1"]["X"]["X_train"]
# to_predict = {
#     "glaz_area_distrib": 0.0,
#     "glazing_area": 0.0,
#     "height": 4.5,
#     "roof_area": 220.5,
#     "surf_area": 686.0,
#     "wall_area": 245.0,
# }
# predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]

# actual = refineRF_PredictionByInterpolation(
#     xdata_names=xdata_names,
#     xdata_numpy=xdata_numpy,
#     to_predict_dict=to_predict,
#     threshold=0.1,
#     predictor=predictor,
# )

# print(actual)


# dict_test = load_object(
#     "/home/jesper/Work/MLDA_app/MLDA/input_data/train_test_dict.sav"
# )
# predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]
# xdata_names = dict_test["N1"]["xdata_names"]
# xdata_numpy = dict_test["N1"]["X"]["X_train"]

# # Exercise
# dict_test = load_object(
#     "/home/jesper/Work/MLDA_app/MLDA/input_data/train_test_dict.sav"
# )
# to_predict = {
#     "glaz_area_distrib": 0,
#     "glazing_area": 0,
#     "height": 3.5,
#     "roof_area": 220,
#     "surf_area": 686,
#     "wall_area": 245,
# }
# xdata_names = dict_test["N1"]["xdata_names"]
# xdata_numpy = dict_test["N1"]["X"]["X_train"]
# df_xdata = pd.DataFrame(data=xdata_numpy, columns=xdata_names)
# print(df_xdata["glazing_area"].value_counts())


# hey = keepIfDataPointsAreComprisingPercentageAboveLimit(
#     data_fraction_limit=0.1,
#     list_of_values=[0.00, 0.10, 0.25, 0.40],
#     percentage_of_range=0.05,
#     xdata=df_xdata,
#     var="glazing_area",
# )

# print(hey)
# actual = makeDFforMultipleRegr(to_predict=to_predict, df_xdata=df_xdata, threshold=0.1)

# print(actual)
# to_predict = {
#     "glaz_area_distrib": 0.0,
#     "glazing_area": 0.0,
#     "height": 3.5,
#     "roof_area": 187.0,
#     "surf_area": 686,
#     "wall_area": 400.0,
# }
# xdata_names = dict_test["N1"]["xdata_names"]
# xdata_numpy = dict_test["N1"]["X"]["X_train"]
# df_xdata = pd.DataFrame(data=xdata_numpy, columns=xdata_names)
# predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]


# test = collectValuesForLinRegr(
#     df_xdata=df_xdata,
#     to_predict=to_predict,
#     threshold=0.08,
#     percent_value_data_range=0.05,
#     data_fraction_limit=0.03,
# )

# predict = refineRF_PredictionByInterpolation(
#     xdata_names=xdata_names,
#     xdata_numpy=xdata_numpy,
#     predictor=predictor,
#     to_predict_dict=to_predict,
#     threshold=0.1,
# )


# print(to_predict)
# print(test)
# print(predict)


# df = pd.DataFrame(data=xdata_numpy, columns=xdata_names)
# df_one_row = df.loc[38, :]
# print(df_one_row.shape)

# df2 = pd.DataFrame(data=[[686, 245, 220.5, 3.5, 0, 0]], columns=xdata_names)
# df_one_row2 = df2.loc[0, :]
# print(df_one_row2.shape)


# x = np.array([686, 245, 220.5, 3.5, 0, 0])


# print(f(x=x, predictor=predictor))
# print(f(x=df_one_row, predictor=predictor))
# # print(df_one_row)

"""For explanation of interpolation method and application examples see end of file."""

import pandas as pd
import numpy as np
from itertools import product
import itertools

from sklearn.linear_model import LinearRegression
from MLDA.miscellaneous.save_and_load_files import load_object, save_object


def makeBooleanMappingOfPredictionArray(
    prediction_array=None,
    min_points=None,
    min_percentage=None,
    criteria_method=None,
    area_frac_to_search_points=None,
    names_bounds=None,
    xdata=None,
    xdata_names=None,
):
    """This function is used to decide whether prediction values should be sent to simple prediction or interpolation prediction 
    is needed. If 'value' is False the variable 'representative' becomes also False, which means that interpolation is needed."""
    pred_array_boolean_map = []
    for prediction in prediction_array:
        representative = True
        for value, var in zip(prediction, xdata_names):
            value = isPointRepresentative(
                point=value,
                min_points=min_points,
                min_percentage=min_percentage,
                criteria_method=criteria_method,
                area_frac_to_search_points=area_frac_to_search_points,
                names_bounds=names_bounds,
                xdata=xdata,
                var=var,
            )
            if not value:
                representative = False
                break
        pred_array_boolean_map.append(representative)
    return pred_array_boolean_map


def basedOnBooleanMappingMakePredArrayForBothSimplePredictionAndInterpolation(
    pred_array_boolean_map=None, prediction_array=None
):
    to_simple_prediction, to_interpolation = [], []
    for boolean, prediction in zip(pred_array_boolean_map, prediction_array):
        if boolean == True:
            to_simple_prediction.append(prediction)
        else:
            to_interpolation.append(prediction)
    return np.array(to_simple_prediction), np.array(to_interpolation)


def mergeY_fromSimplePredictionAndFromInterpolation(
    Y_simple_prediction=None, Y_interpolation=None, pred_array_boolean_map=None
):
    Y_simple_prediction, Y_interpolation = (
        Y_simple_prediction.tolist(),
        Y_interpolation.tolist(),
    )
    Y_array = []
    for boolean in pred_array_boolean_map:
        if boolean == True:
            Y_array.append(Y_simple_prediction.pop(0))
        elif boolean == False:
            Y_array.append(Y_interpolation.pop(0))
    return Y_array


def returnBooleanBasedOnCriteria(
    criteria_method=None,
    min_points=None,
    min_percentage=None,
    count_points=None,
    xdata=None,
    var=None,
):
    if criteria_method == "default":
        min_points = 2
        if count_points >= min_points:
            return True
    if criteria_method == "num_points":
        if count_points >= min_points:
            return True
    if criteria_method == "points_percentage":
        count_points_percentage = count_points / xdata[var].count()
        if count_points_percentage >= min_percentage:
            return True


def isPointRepresentative(
    point=None,
    min_points=None,
    min_percentage=None,
    criteria_method=None,
    area_frac_to_search_points=None,
    names_bounds=None,
    xdata=None,
    var=None,
):
    """Function that returns True or False given a point. 
    Approach: Given a point we want to be able to say if this point is close enough to other points within a
    certain specified area, which is calc by using the variable 'area_frac_to_search_points'. The 'close enough' condition 
    can be either number of neighbor points in the neighbor area or the percentage these neighbor points comprises 
    of the total number of points. If enough points is inside specified neighbor area, function returns True, else False.
    """
    # First we find the total area (or neighbor area), i.e. both sides
    area_var = (
        names_bounds[var][1] - names_bounds[var][0]
    ) * area_frac_to_search_points
    # Now calc how far to each side we can collect points
    lower_limit, upper_limit = point - area_var / 2, point + area_var / 2
    # Now finding number of points in that area, inside lower and upper limit
    count_points = xdata[var][
        (xdata[var] < upper_limit) & (xdata[var] > lower_limit)
    ].count()

    # Now checking if number of points found is higher or lower
    # than specified condition.
    boolean = returnBooleanBasedOnCriteria(
        criteria_method=criteria_method,
        min_points=min_points,
        min_percentage=min_percentage,
        count_points=count_points,
        xdata=xdata,
        var=var,
    )
    if boolean:
        return True
    else:
        return False


def returnFirstPointFromListThatPassesCriteria(
    list_of_points=None,
    var=None,
    names_bounds=None,
    xdata=None,
    criteria_method=None,
    min_points=None,
    min_percentage=None,
    area_frac_to_search_points=None,
):
    """criteria_method can either be num_points, points_percentage or default. Default is num_points."""
    for point in list_of_points:
        boolean = isPointRepresentative(
            point=point,
            min_points=min_points,
            min_percentage=min_percentage,
            criteria_method=criteria_method,
            area_frac_to_search_points=area_frac_to_search_points,
            names_bounds=names_bounds,
            xdata=xdata,
            var=var,
        )
        if boolean:
            return point


def findLowerAndHigherPoints(xdata=None, var=None, value=None):
    """Given xdata, var and value this function returns the closest
    lower and upper points. The points may be equal to the value.
    Input:
    xdata: dataframe
    var: string
    value: float number
    Output:
    two-value tuple with closest lower and closest higher point
    """
    points_lower = xdata[var][xdata[var] <= value]
    points_upper = xdata[var][xdata[var] >= value]
    points_lower.drop_duplicates(inplace=True)
    points_upper.drop_duplicates(inplace=True)
    points_lower.sort_values(ascending=False, inplace=True)
    points_upper.sort_values(ascending=True, inplace=True)
    points_lower, points_upper = (points_lower.to_numpy(), points_upper.to_numpy())
    return points_lower, points_upper


def collectInterpolationPoints(
    prediction_array=None,
    xdata_names=None,
    names_bounds=None,
    xdata=None,
    criteria_method=None,
    min_points=None,
    min_percentage=None,
    area_frac_to_search_points=None,
):
    """
    Function that collects array with points to be used for interpolatiæon calc.
    Approach: for each prediction (i.e. for each array of values in prediction_array) we go through each value and finds
    the closest lower and closest higher point for this value. These points should then be used for calculation of the
    interpolated prediction value. If both the closest lower and closest higher point is equal to the value - which is 
    the case when we have a value in our prediction that is a representative training data point, which is often the 
    case - we use this value for both the closest lower and the closest higher point.
    
    Input:
    prediction_array: array of predictions,
    xdata_names: dataframe index,
    names_bounds: dict
    xdata: dataframe
    criteria_method: string,
    min_points: integer,
    min_percentage: fraction/float,
    area_frac_to_search_points: fraction/float,

    Output:
    list of array of bounds (list of numpy arrays): ex: [[[3, 4], [10, 20], [100, 110]], [[5, 6], [10, 20], [200, 210]]]
    """
    # list to collect lists of prediction bounds - e.g. a list like this:
    # [[[3, 4], [10, 20], [100, 110]], [[5, 6], [10, 20], [200, 210]]]
    lower_upper_array = []
    for prediction in prediction_array:
        lower_upper = []  # list to collect prediction bounds
        # e.g. lower_upper = [[3, 4], [10, 20], [100, 110]]
        # Now for each value in each prediction we find the lower/higher points
        for value, var in zip(prediction, xdata_names):
            points_lower, points_upper = findLowerAndHigherPoints(
                xdata=xdata, var=var, value=value
            )
            lower_point = returnFirstPointFromListThatPassesCriteria(
                list_of_points=points_lower,
                var=var,
                names_bounds=names_bounds,
                xdata=xdata,
                criteria_method=criteria_method,
                min_points=min_points,
                area_frac_to_search_points=area_frac_to_search_points,
            )
            upper_point = returnFirstPointFromListThatPassesCriteria(
                list_of_points=points_upper,
                var=var,
                names_bounds=names_bounds,
                xdata=xdata,
                criteria_method=criteria_method,
                min_points=min_points,
                area_frac_to_search_points=area_frac_to_search_points,
            )
            if lower_point == None:
                lower_point = names_bounds[var][0]
            if upper_point == None:
                upper_point = names_bounds[var][1]
            lower_upper.append([lower_point, upper_point])
        lower_upper_array.append(lower_upper)
    return lower_upper_array


# # Input
# ML_dictEN2 = load_object("/home/jesper/Work/MLDA_app/MLDA/jupyter_ML/ML_dictEN2.sav")
# xdata_names = ML_dictEN2["xdata_names"]
# xdata = ML_dictEN2["df_data"][xdata_names]

# names_bounds = ML_dictEN2["X_bounds"]
# rf_predictor = ML_dictEN2["Y"]["heat_load"]["pred"]["forest"]["predictor"]

# # print(ML_dictEN2.keys())
# print(xdata.head(40))

# values = [[416.5, 147.0, 4.0, 0.0, 0.0], [416.5, 137.0, 4.0, 0.0, 0.0]]


# print(
#     collectInterpolationPoints(
#         prediction_array=values,
#         xdata_names=xdata_names,
#         xdata=xdata,
#         names_bounds=names_bounds,
#         criteria_method="default",
#         area_frac_to_search_points=1,
#     )
# )


# Functions


def f(x=None, predictor=None):
    """Function that takes an array and a predictor and returns an prediction array. 
    Approach: if not already a numpy array then make it one. And if input is a single row
    then reshape input.
    Input: 
    x: numpy array, pandas df/series or python list
    predictor: sklearn predictor
    """
    if len(x) == 0:
        return np.array([])
    if not hasattr(x, "reshape"):
        x = np.array(x)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    return predictor.predict(x)


def makeXarrayForInterpolationPredictions(bounds_array=None):
    """
    sum_list has all lower/upper points to make interpolation for a given prediction.
    The structure is the following: first prediction at first feature has its lower points
    repeated 2**num_feat/(2**index+1) times, followed by its upper points also repeated
    2**num_feat/(2**index+1) times, which make a total of 16+16 points for first index in ex above.
    Next is feature number 2 in first prediction: again its lower points repeated 2**num_feat/(2**index+1)
    times, followed by its upper points also repeated 2**num_feat/(2**index+1) times, which make a total
    of 8+8+8+8 points for second index in ex above. And so on.. This structure gives all combinations of
    lower/upper points making interpolation possible

    
    Function that takes an array of bounds and returns a long array with number of rows: num_prediction*2**num_feat,
    and each row has a length of num_feat. So the returned array has the shape(num_prediction*2**num_feat, num_feat). 
    For a 2 predictions task like this:
    predictions_values = [[411.5, 143.0, 4.0, 0.2, 0.2], [410.5, 117.0, 4.0, 0.0, 0.0]]
    the upper_lower_points or bounds_array could look like this:
    bounds_array = [[[367.5, 416.5], [122.5, 147.0], [4, 4], [0.1, 0.25], [0, 1]], [[367.5, 416.5], [110.25, 122.5], [4, 4], [0.0, 0.0], [0, 0]]]
    Now this function converts the bounds_array to one long array of rows, where each row is ready for a RF prediction. Note all rows for prediction
    are one after another in the returned array. In the above example we have to 5 feature predictions which gives: 2*2**5 = 64 rows. Now these rows
    are reshaped later on. The reshaping is done so each prediction gets its own columns. 
    Input: 
    bounds_array (upper_lower_points): list
    Output: 
    """
    # From the first entry in bounds_array we calc num_rows and num_feat
    num_rows, num_features = 2 ** len(bounds_array[0]), len(bounds_array[0])
    sum_sum_list = (
        []
    )  # list to collect all lower_upper combinations for each prediction
    for entry in bounds_array:
        # For ex: 1st entry would be: [[367.5, 416.5], [122.5, 147.0], [4, 4], [0.1, 0.25], [0, 1]]
        # in the above example
        sum_list = (
            []
        )  # list to collect all lower_upper combinations for each index, when all
        # index is looped through it is appended to sum_sum_list
        for index, item in enumerate(entry):
            # print("index:", index)
            # for each index in entry we....
            # index 0 is: [367.5, 416.5]
            collect_list = (
                []
            )  # list to collect lower_upper for each i in range(2 ** index)
            for i in range(2 ** index):
                # and for index 0 i would be in range(1) since 2**0 = 1
                # so for index 0 we would do what is in the loop one time
                list_ = [entry[index][0]] * int(num_rows / 2 ** (index + 1)) + [
                    entry[index][1]
                ] * int(num_rows / 2 ** (index + 1))
                # for index 0 list_ gives: [367.5, 367.5,...16 times, and then 416.5, 416.5.. 16 times] one time
                # for index 2 range(2 ** index) gives range(4) and list_ gives: [4, 4, 4, 4, 4, 4, 4, 4] four times
                # all have a total number of points equal to 2**num_feat, in this ex: 32
                # print("i:", i)
                # print("list_:", list_)
                # Now append list_ to collect_list for each i in range(2**index)
                collect_list.append(list_)
            #     print()
            #     print("collect_list:", collect_list)
            # print("collect_list", collect_list)
            # Since collect_list may have different numbers of lists depending
            # on i in range(2**index) we now flatten it, so all collect_lists
            # have same size and shape
            flatten_list = list(itertools.chain.from_iterable(collect_list))
            # Then for each flattened collect_list with all lower_upper points for given
            # index we now add each of these to sum_list
            # print()
            # print("flatten_list", flatten_list)
            sum_list.append(flatten_list)
            # print()
            # print("sum_list", sum_list)
            # print()

        # now to make these values in sum_list to be digestable for the RF predictor we take the long list for each
        # feature with all lower/upper combinations (5 long lists in the ex above) and reshape these to small lists
        # with n features in each (again 5 in the ex above). So we go from sum_list looking like:
        # sum_list= [[367.5, 367.5,....., 367.5, 416.5, 416.5, .. 416.5], [122.5, ...122.5,, 147.0, ..147.0, 122.5,... 122.5, 147.0, ..147.0], [4, ... 4], [0.1, 0.1, 0.25, 0.25, 0.1, 0.1, 0.25, 0.25, 0.1, 0.1, 0.25, 0.25, 0.1, 0.1, 0.25, 0.25, 0.1, 0.1, 0.25, 0.25, 0.1, 0.1, 0.25, 0.25, 0.1, 0.1, 0.25, 0.25, 0.1, 0.1, 0.25, 0.25], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]
        # to sum_list_T = [[367.5, 122.5, 4, 0.1, 0], [367.5, 122.5, 4, 0.1, 1], [367.5, 122.5, 4, 0.25, 0], [367.5, 122.5, 4, 0.25, 1], [367.5, 122.5, 4, 0.1, 0], [367.5, 122.5, 4, 0.1, 1], [367.5, 122.5, 4, 0.25, 0], [367.5, 122.5, 4, 0.25, 1], [367.5, 147.0, 4, 0.1, 0], [367.5, 147.0, 4, 0.1, 1], [367.5, 147.0, 4, 0.25, 0], [367.5, 147.0, 4, 0.25, 1], [367.5, 147.0, 4, 0.1, 0], [367.5, 147.0, 4, 0.1, 1], [367.5, 147.0, 4, 0.25, 0], [367.5, 147.0, 4, 0.25, 1], [416.5, 122.5, 4, 0.1, 0], [416.5, 122.5, 4, 0.1, 1], [416.5, 122.5, 4, 0.25, 0], [416.5, 122.5, 4, 0.25, 1], [416.5, 122.5, 4, 0.1, 0], [416.5, 122.5, 4, 0.1, 1], [416.5, 122.5, 4, 0.25, 0], [416.5, 122.5, 4, 0.25, 1], [416.5, 147.0, 4, 0.1, 0], [416.5, 147.0, 4, 0.1, 1], [416.5, 147.0, 4, 0.25, 0], [416.5, 147.0, 4, 0.25, 1], [416.5, 147.0, 4, 0.1, 0], [416.5, 147.0, 4, 0.1, 1], [416.5, 147.0, 4, 0.25, 0], [416.5, 147.0, 4, 0.25, 1]]
        sum_list_T = list(map(list, zip(*sum_list)))
        # print("sum_list_T", sum_list_T)
        # Now for each entry/prediction we append all RF-prediction-ready points for this entry to sum_sum_list
        sum_sum_list.append(sum_list_T)
    # sum_sum_list now contains all info to make interpolation using RF-predictior
    sum_sum_list = np.array(sum_sum_list)
    # print()
    # print("sum_sum_list", sum_sum_list)
    # print()
    # After the appending process we have to reshape sum_sum_list, so instead of having a
    # shape(num_pred, 2**num_feat, num_feat), in ex above: shape(2, 32, 5), we reshape to:
    # reshape(num_predictions * 2**num_feat, num_feat), in ex above: shape(2 * 32, 5)
    num_predictions, rows = sum_sum_list.shape[0], sum_sum_list.shape[1]
    sum_sum_list = sum_sum_list.reshape(num_predictions * rows, num_features)
    return sum_sum_list


# bounds_array = [
#     [[367.5, 416.5], [122.5, 147.0], [4, 4], [0.1, 0.25], [0, 1]],
#     [[367.5, 416.5], [110.25, 122.5], [4, 4], [0.0, 0.0], [0, 0]],
# ]
# sum_sum_list = makeXarrayForInterpolationPredictions(bounds_array=bounds_array)

# Xdelta:
def makeXdeltaListAndXAboveLowbounds(bounds_array=None, prediction_array=None):
    Xdelta_list, Xlowbounds = [], []
    for bounds in bounds_array:
        Xdelta = [bound[1] - bound[0] for bound in bounds]
        Xlowbound = [bound[0] for bound in bounds]
        Xdelta_list.append(Xdelta)
        Xlowbounds.append(Xlowbound)
    # Tranposing prediction_array
    prediction_array_T = np.transpose(np.array(prediction_array))
    Xdelta_list, Xlowbounds = np.array(Xdelta_list), np.array(Xlowbounds)
    # Now replace all 0's in Xdelta_list with 1's to avoid dividing with 0 later on
    Xdelta_list[Xdelta_list == 0] = 1
    # Tranposing Xdelta_list:
    Xdelta_list, Xlowbounds = np.transpose(Xdelta_list), np.transpose(Xlowbounds)
    # Calc Xabove_lowbounds, i.e. how far the prediction value is from lowbound
    Xabove_lowbounds = prediction_array_T - Xlowbounds
    return Xdelta_list, Xabove_lowbounds


# predict, array_bounds = (
#     [[3.5, 14, 59], [3, 14, 59]],
#     [[[3, 4], [10, 20], [50, 60]], [[3, 3], [10, 20], [50, 60]]],
# )
# Xdelta_list, Xabove_lowbounds = makeXdeltaListAndXAboveLowbounds(
#     bounds_array=array_bounds, prediction_array=predict
# )
# print(Xdelta_list)
# print()
# print(Xabove_lowbounds)


def makeInterpolation(Xdelta_list=None, Xabove_lowbounds=None, Yall_lower_upper=None):
    """
    In this function we make the interpolation calculations.

    Input: 
    bounds_array: type: list
    prediction_array: type: list or numpy array. Array to make prediction on. 
    predictor: type: see randomForestInterpolation

    Output:
    Array with interpolated prediction values.
    """
    for rowXdelta, rowXabove in zip(Xdelta_list, Xabove_lowbounds):
        Y_1st_half, Y_2nd_half = (
            Yall_lower_upper[0 : int(len(Yall_lower_upper) / 2)],
            Yall_lower_upper[int(len(Yall_lower_upper) / 2) : len(Yall_lower_upper)],
        )
        Ydelta = np.array(Y_2nd_half) - np.array(Y_1st_half)
        # print("Ydelta:", Ydelta)
        # print()
        # print("rowXdelta", rowXdelta)
        # print()
        # print("Ydelta/rowXdelta:", Ydelta / rowXdelta)

        Ynext = Y_1st_half + (Ydelta / rowXdelta) * rowXabove
        # print("Ynext:", Ynext)
        Yall_lower_upper = Ynext
    return Yall_lower_upper


# tist = makeInterpolation(
#     Xdelta_list=[[1, 1], [10, 10], [10, 10]],
#     Xabove_lowbounds=[[0.5, 0.1], [4, 7], [1, 6]],
#     Yall_lower_upper=[
#         [29, 23],
#         [31, 21],
#         [30, 20],
#         [29, 23],
#         [37, 26],
#         [32, 25],
#         [30, 23],
#         [30, 27],
#     ],
# )


def prepareForInterpolation(bounds_array=None, prediction_array=None, predictor=None):
    """
    In this function we retrieve the relevant data for the interpolation calculations.
    
    Input: 
    bounds_array: type: list
    prediction_array: type: list or numpy array. Array to make prediction on. 
    predictor: type: see randomForestInterpolation

    Output: 
    Xdelta_list, Xabove_lowbounds, Yall_lower_upper: type: all are numpy array. 
    """

    # From the bounds_array and prediction_array we find Xdelta_list, Xabove_lowbounds
    # that is the difference in lower and upper point, and how much the prediction value
    # is above the lower point.
    Xdelta_list, Xabove_lowbounds = makeXdeltaListAndXAboveLowbounds(
        bounds_array=bounds_array, prediction_array=prediction_array
    )
    # print("Xdelta_list:", Xdelta_list)
    # print()
    # print("Xabove_lowbounds", Xabove_lowbounds)
    num_predictions, num_features = len(prediction_array), len(prediction_array[0])

    # So far we have the all the closest lower and closest higher points for all
    # our prediction arrays. Now we need to convert all these sets of points (lower/higher)
    # to one long array with a row length of num_predictions*2**num_feat and each row can be
    # calc by the RF_predictor.
    Xarray_for_interpolation = makeXarrayForInterpolationPredictions(
        bounds_array=bounds_array
    )
    # print()
    # print("Xarray_for_interpolation_shape", np.shape(Xarray_for_interpolation))
    # print("Xarray_for_interpolation", Xarray_for_interpolation)
    # print()

    Yall_lower_upper = f(x=Xarray_for_interpolation, predictor=predictor)
    # print("Yall_lower_upper:", Yall_lower_upper)
    # print()
    # Number of columns are equal to num_predictions
    Yall_lower_upper = Yall_lower_upper.reshape(num_predictions, 2 ** num_features)
    # print("Yall_lower_upper:", Yall_lower_upper)
    # print()
    Yall_lower_upper = np.transpose(Yall_lower_upper)
    # print("shape(Y):", np.shape(Yall_lower_upper))
    # print("Yall_lower_upper:", Yall_lower_upper)
    # print()
    return Xdelta_list, Xabove_lowbounds, Yall_lower_upper


def randomForestInterpolation(
    prediction_array=None,
    xdata_names=None,
    names_bounds=None,
    xdata=None,
    RF_predictor=None,
    criteria_method=None,
    min_points=None,
    min_percentage=None,
    area_frac_to_search_points=None,
):
    """
    In the function makeBooleanMappingOfPredictionArray we find out if our prediction point contains only 
    representative points, and if not we use randomForestInterpolation to make a good prediction as an alternative.
    
    Input: 
    prediction_array: type: list or numpy array. Array to make prediction on. 
    xdata_names: type: pandas.core.indexes.base.Index
    names_bounds: type: dict
    xdata: type: pandas dataframe
    RF_predictor: type: sklearn.ensemble.forest.RandomForestRegressor
    criteria_method, min_points, min_percentage, area_frac_to_search_points: see randomForestPredictorWithInterpolation

    Output: type: numpy array. Array with interpolated prediction values.
    """
    # print("type(xdata_names", type(RF_predictor))
    if len(prediction_array) == 0:
        return np.array([])
    # Collect array of bounds for interpolation
    bounds_array = collectInterpolationPoints(
        prediction_array=prediction_array,
        xdata_names=xdata_names,
        names_bounds=names_bounds,
        xdata=xdata,
        criteria_method=criteria_method,
        area_frac_to_search_points=area_frac_to_search_points,
    )
    # Prepare for interpolation calc
    Xdelta_list, Xabove_lowbounds, Yall_lower_upper = prepareForInterpolation(
        bounds_array=bounds_array,
        prediction_array=prediction_array,
        predictor=RF_predictor,
    )
    # Now interpolation calc
    Y = makeInterpolation(
        Xdelta_list=Xdelta_list,
        Xabove_lowbounds=Xabove_lowbounds,
        Yall_lower_upper=Yall_lower_upper,
    )
    return Y[0]


def mappingSplittingAndMergingWhenInterpolationTrue(
    prediction_array=None,
    criteria_method="default",
    criteria_method_boolean_map="num_points",
    min_points=None,
    min_points_boolean_map=2,
    min_percentage=None,
    min_percentage_boolean_map=None,
    area_frac_to_search_points=0.05,
    area_frac_to_search_points_boolean_map=0.02,
    names_bounds=None,
    xdata=None,
    xdata_names=None,
    RF_predictor=None,
):
    # Make mapping of required interpolation and not required interpolation
    pred_array_boolean_map = makeBooleanMappingOfPredictionArray(
        prediction_array=prediction_array,
        min_points=min_points_boolean_map,
        min_percentage=min_percentage_boolean_map,
        criteria_method=criteria_method_boolean_map,
        area_frac_to_search_points=area_frac_to_search_points_boolean_map,
        names_bounds=names_bounds,
        xdata=xdata,
        xdata_names=xdata_names,
    )
    # Make arrays to RF_prediction and to RF_interpolation
    to_RF_prediction, to_RF_interpolation = basedOnBooleanMappingMakePredArrayForBothSimplePredictionAndInterpolation(
        pred_array_boolean_map=pred_array_boolean_map, prediction_array=prediction_array
    )

    # Sent to RF_prediction and to RF_interpolation
    Y_RF_prediction = f(x=to_RF_prediction, predictor=RF_predictor)
    Y_RF_interpolation = randomForestInterpolation(
        prediction_array=to_RF_interpolation,
        xdata_names=xdata_names,
        names_bounds=names_bounds,
        xdata=xdata,
        RF_predictor=RF_predictor,
        criteria_method=criteria_method,
        min_points=min_points,
        min_percentage=min_percentage,
        area_frac_to_search_points=area_frac_to_search_points,
    )
    # Merge RF_prediction and RF_interpolation in accordance with to pred_array_boolean_map
    Y = mergeY_fromSimplePredictionAndFromInterpolation(
        Y_simple_prediction=Y_RF_prediction,
        Y_interpolation=Y_RF_interpolation,
        pred_array_boolean_map=pred_array_boolean_map,
    )
    return np.array(Y)


def randomForestPredictorWithInterpolation(
    interpolation=True,
    prediction_array=None,
    ML_dict=None,
    yvar=None,
    criteria_method="default",
    criteria_method_boolean_map="num_points",
    min_points=None,
    min_points_boolean_map=2,
    min_percentage=None,
    min_percentage_boolean_map=None,
    area_frac_to_search_points=0.05,
    area_frac_to_search_points_boolean_map=0.02,
):
    """
    Function that besides making normal random forest predictions also makes interpolations between random forest predictions. 
    This option is valuable if you want to predict values in areas where the numbers of training points are lean or missing.

    Input: 
    prediction_array: type: list or numpy array. Array to make prediction on. 
    ML_dict: type: dict. Name of ML_dict, e.g.: ML_dictEN2. 
    yvar: type: str. Name of dependent variable to predict. 
    criteria_method: type: str. Method to use in randomForestInterpolation, when choosing points for interpolation. Options are: 'default', 'num_points' or 
    'points_percentage'. 'default' is 'num_points' with min_points set to 2. If choosing 'num_points' or 'points_percentage' also choose value for
    'min_points' or 'min_percentage', respectively. 
    criteria_method_boolean_map: type: str. method to use in makeBooleanMappingOfPredictionArray, to check each prediction in prediction_array to see if 
    interpolation is required or not. Options are: as for criteria_method above. 
    min_points: type: int. Related to criteria_method when choosing 'num_points'. 
    min_points_boolean_map: type: int. Related to criteria_method_boolean_map when choosing 'num_points'. 
    min_percentage: type: float. Related to criteria_method when choosing 'points_percentage'. 
    min_percentage_boolean_map: type: float. Related to criteria_method_boolean_map when choosing 'points_percentage'.
    area_frac_to_search_points: type: float. Used in isPointRepresentative to search each side of a point to find how many neighbor points are within that
    area. The lower we set this value the more restrict is the criteria of representativeness since this means we only choose points that have a lot of 
    neighbors within a small area, else we reject it. 
    area_frac_to_search_points_boolean_map: type: float. In makeBooleanMappingOfPredictionArray area_frac_to_search_points_boolean_map is used as a fraction
    value when deciding whether interpolation is required or not. The lower the fraction limit the less the allowed difference between a given feature value 
    and its corresponding feature values in the training data set, before interpolation is required. (screening value)
    bounds_array (upper_lower_points): : type: list
    
    Output: type: numpy array. Array of prediction values.
    """
    xdata_names, df_data = ML_dict["xdata_names"], ML_dict["df_data"]
    xdata, names_bounds = df_data[xdata_names], ML_dict["X_bounds"]
    RF_predictor = ML_dict["Y"][yvar]["pred"]["forest"]["predictor"]

    if interpolation == True:
        Y = mappingSplittingAndMergingWhenInterpolationTrue(
            prediction_array=prediction_array,
            criteria_method=criteria_method,
            criteria_method_boolean_map=criteria_method_boolean_map,
            min_points=min_points,
            min_points_boolean_map=min_points_boolean_map,
            min_percentage=min_percentage,
            min_percentage_boolean_map=min_percentage_boolean_map,
            area_frac_to_search_points=area_frac_to_search_points,
            area_frac_to_search_points_boolean_map=area_frac_to_search_points_boolean_map,
            names_bounds=names_bounds,
            xdata=xdata,
            xdata_names=xdata_names,
            RF_predictor=RF_predictor,
        )
    elif interpolation == False:
        Y = f(x=prediction_array, predictor=RF_predictor)
    return Y


# Input Constructed data - ML_dictCON1
# ML_dictCON1 = load_object("/home/jesper/Work/MLDA_app/MLDA/jupyter_ML/ML_dictCON1.sav")
# values = [[3.0, 20.0, 70], [3.8, 23.5, 65]]
# predict_y = randomForestPredictorWithInterpolation(
#     interpolation=True, prediction_array=values, ML_dict=ML_dictCON1, yvar="Y"
# )
# print(predict_y)  # False : [30.89396667 31.4089    ]   True: [30.89396667 33.494675  ]


# xdata_names = ML_dictEN2["xdata_names"]
# xdata = ML_dictEN2["df_data"][xdata_names]
# # names_bounds = ML_dictEN2["X_bounds"]
# # rf_predictor = ML_dictEN2["Y"]["heat_load"]["pred"]["forest"]["predictor"]

# # print(ML_dictEN2.keys())
# # print(xdata.head(40))

# values = [[416.5, 122.5, 4.0, 0.0, 0.0], [416.5, 134.75, 4.0, 0.0, 0.0]]
# # print(f(values, rf_predictor))  # 27.32366
# Y = randomForestPredictorWithInterpolation(
#     prediction_array=values,
#     ML_dict=ML_dictEN2,
#     yvar="heat_load",
#     min_points_boolean_map=2,
#     area_frac_to_search_points=1,
#     area_frac_to_search_points_boolean_map=0.04,
# )
# print(Y)
# Y_interp: 20.85940375
# print(bounds_array)
# names_values = {
#     "wall_area": 245,
#     "roof_area": 120.5,
#     "orientation": 4,
#     "glazing_area": 0,
#     "glaz_area_distrib": 0,
# }
# roof_area=[110.25, 122.5, 147.0, 220.5]
# wall_area=[245.0, 269.5, 294.0, 318.5, 343.0, 367.5, 416.5]


"""
Let's see how randomForestPredictorWithInterpolation works by going through an example:
To make a prediction we have to pass in data for prediction_array, ML_dict and yvar. prediction_array contains feature values that
we want to make prediction on (it may contain a single set or many sets of feature values). ML_dict is the dict with training data
etc. yvar is the dependent var - the variable that we want to predict.

The overall structure of randomForestPredictorWithInterpolation is this:

1) make a mapping of the items in prediction_array, that needs interpolation and not (mapping is stores in pred_array_boolean_map)
2) based on pred_array_boolean_map make a list that should be send directly to prediction (without interpolation) and a list that 
should have interpolation (to_RF_prediction and to_RF_interpolation, respectively)
3) Sent both RF_prediction and RF_interpolation to be processed and we get Y_RF_prediction and Y_RF_interpolation in return, 
respectively  
4) Merge Y_RF_prediction and Y_RF_interpolation in accordance with to pred_array_boolean_map

The example is divided into these 4 points.

1) 

collectInterpolationPoints
Function that collects array with points to be used for interpolatiæon calc.
Approach: for each prediction (i.e. for each array of values in prediction_array) we go through each value and finds
the closest lower and closest higher point for this value. These points should then be used for calculation of the
interpolated prediction value. If both the closest lower and closest higher point is equal to the value - which is 
the case when we have a value in our prediction that is a representative training data point, which is often the 
case - we use this value for both the closest lower and the closest higher point.


Example:
Let's see how randomForestInterpolation works by showing an example: say we pass this input (which is a single array) 
to our prediction_array: [[3.5, 14, 51]] since it has some unrepresentative points. Let's say all points are unrepresentative 
points. Now to make a prediction we need some representative points to interpolate between. For that we use the 
collectInterpolationPoints which, let's say give us these values: [[3, 10, 50], [4, 20, 60]] (from the training data). 
So now we have lower and upper points for all the 3 features for in [3.5, 14, 51]. The next step is to prepare for 
interpolation, where we use the prepareForInterpolation function. prepareForInterpolation first finds Xdelta_list, 
Xabove_lowbounds by using makeXdeltaListAndXAboveLowbounds. For this ex Xdelta_list=[[1], [10], [10]] since this is the difference
between the lower and upper values shown above, the Xabove_lowbounds = [[.5], [4], [1]] since this is the difference between the 
lower and the prediction values shown above. 
Now the next part is where the key manipulation of data takes place. By using makeXarrayForInterpolationPredictions we now
take the lower/upper values and finds all posible combinations of these. For this ex the result is:
combinations = [[3, 10, 50], [3, 10, 60], [3, 20, 50], [3, 20, 60], [4, 10, 50], [4, 10, 60], [4, 20, 50], [4, 20, 60]]
Notice that the result is a list with length of 2**num_features (actually it is 2**num_features * num_predictions). 
And notice how we for the first four list items keep the first feature at 3 and for the next four keep it at 4, and
for the second feature the pattern is shifting for eac two values and so on. Now after this we have to make a some
transposing and reshaping in order to be able to pass it to the random forest predictor. 
When passed to the random forest predictor we get prediction values for all combinations of the 8 items list above. 
Say we get these 8 values: prediction_result = [29,31,30,29,37,32,30,30], where 29 is from [3, 10, 50], 31 from [3, 10, 60]
and so on. 

Now the final step is to make the interpolation between these 8 prediction values, where we use the makeInterpolation function. 
We do that by first looking at the first feature value in our input which is 3.5. Now we do not have any prediction result 
when the first feature has the value 3.5. But we have prediction results when the first feature is 3 and 4. The 1st and 5th 
item in 'combinations' above is [3, 10, 50] and [4, 10, 50] and the corresponding prediction results are 29 and 37. Note the 
two other features are the same in 1st and 5th item (i.e. 10 and 50, respectively). So the difference in prediction results 
(the difference between 29 and 37) is then only due to the difference in the first feature (i.e. between 3 and 4, respectively). 
So now we can shrink the combinations list above by saying we now go from 3 to 3.5 for the first feature. By doing this 
we have increase the prediction result from 29 to half way up to 37 (which is 33), since 37 is when the first feature is 4. 
So we can replace the 1st and 5th item in 'combinations' above with [3.5, 10, 50] and replace 29 and 37 with 33. Doing this 
for all 8 items in 'combinations' gives this:

combinations = [[3.5, 10, 50], [3.5, 10, 60], [3.5, 20, 50], [3.5, 20, 60]] 
and the corresponding prediction results: prediction_result = [37, 31.5, 30, 29.5]

Next we do the same for 2nd feature where we have prediction results for 10 and 20 and want to find for 14. 
combinations = [[3.5, 14, 50], [3.5, 14, 60]];  prediction_result = [31.8, 30.7]
And finally for the last feature:
combinations = [[3.5, 14, 51];  prediction_result = [31.7]

End example.
sum_list has all lower/upper points to make interpolation for a given prediction.
The structure is the following: first prediction at first feature has its lower points
repeated 2**num_feat/(2**index+1) times, followed by its upper points also repeated
2**num_feat/(2**index+1) times, which make a total of 16+16 points for first index in ex above.
Next is feature number 2 in first prediction: again its lower points repeated 2**num_feat/(2**index+1)
times, followed by its upper points also repeated 2**num_feat/(2**index+1) times, which make a total
of 8+8+8+8 points for second index in ex above. And so on.. This structure gives all combinations of
lower/upper points making interpolation possible


Function that takes an array of bounds and returns a long array with number of rows: num_prediction*2**num_feat,
and each row has a length of num_feat. So the returned array has the shape(num_prediction*2**num_feat, num_feat). 
For a 2 predictions task like this:
predictions_values = [[411.5, 143.0, 4.0, 0.2, 0.2], [410.5, 117.0, 4.0, 0.0, 0.0]]
the upper_lower_points or bounds_array could look like this:
bounds_array = [[[367.5, 416.5], [122.5, 147.0], [4, 4], [0.1, 0.25], [0, 1]], [[367.5, 416.5], [110.25, 122.5], [4, 4], [0.0, 0.0], [0, 0]]]
Now this function converts the bounds_array to one long array of rows, where each row is ready for a RF prediction. Note all rows for prediction
are one after another in the returned array. In the above example we have to 5 feature predictions which gives: 2*2**5 = 64 rows. Now these rows
are reshaped later on. The reshaping is done so each prediction gets its own columns. 



Prepare for interpolation:
Xdelta_list and Xabove_lowbounds is found by makeXdeltaListAndXAboveLowbounds. 
And Xarray_for_interpolation by makeXarrayForInterpolationPredictions.

Preparing data for interpolation calculations:
As explained in the text for randomForestInterpolation the Xarray_for_interpolation contains sections of rows with the shape: 
(num_predictions * 2**num_feat, num_feat). A section of rows for each prediction to make. And each row in these sections
is a unique combination of lower and upper points for this particular prediction. And as also mentioned in the text the 
sections lay on top of each other - first one first and so forth. As an example for a 2-value prediction array like this:
prediction_array: [[3.5, 14, 51], [2.1, 27, 76]] the Xarray_for_interpolation may contain these values:

[[3, 10, 50], [3, 10, 60], [3, 20, 50], [3, 20, 60], [4, 10, 50], [4, 10, 60], [4, 20, 50], [4, 20, 60], 
[2, 20, 70], [2, 20, 80], [2, 30, 70], [2, 30, 80], [3, 20, 70], [3, 20, 80], [3, 30, 70], [3, 30, 80]]

first 8 items (first line) is first section of rows, i.e. the unique combination of lower and upper points for the prediction
[3.5, 14, 51]. 

Now passing the values in Xarray_for_interpolation to the predictor then gives num_predictions * 2**num_feat prediction values 
(Yall_lower_upper). In our ex the predictor may return these values:
Yall_lower_upper =  [[29], [31], [30], [29], [37], [32], [30], [30], [23], [21], [20], [23], [26], [25], [23], [27]]
Again, note that the first 8 items are prediction values generated from unique combination of lower and upper points for the
prediction of [3.5, 14, 51] that we have to make.  

Now in order to make the next calculations more efficient we we reshape and transpose Yall_lower_upper so Yall_lower_upper 
now have this shape: (2**num_feat, num_predictions). Which given the ex. above would give a 8 rows and 2 columns shape:
Yall_lower_upper = [[29, 23], [31, 21], [30, 20], [29, 23], [37, 26], [32, 25], [30, 23], [30, 27]]

For this ex the Xdelta_list and Xabove_lowbounds would be:

Xdelta_list = [[1, 1], [10, 10], [10, 10]]; Xabove_lowbounds = [[.5, .1], [4, 7], [1, 6]]



Interpolation calculations:
Now to see how makeInterpolation works we will continue with our example in prepareForInterpolation where we had ended up with
these values:

prediction_array = [[3.5, 14, 51], [2.1, 27, 76]]
Xarray_for_interpolation = 
                [[3, 10, 50], [3, 10, 60], [3, 20, 50], [3, 20, 60], [4, 10, 50], [4, 10, 60], [4, 20, 50], [4, 20, 60], 
                [2, 20, 70], [2, 20, 80], [2, 30, 70], [2, 30, 80], [3, 20, 70], [3, 20, 80], [3, 30, 70], [3, 30, 80]]
Yall_lower_upper = [[29, 23], [31, 21], [30, 20], [29, 23], [37, 26], [32, 25], [30, 23], [30, 27]]
Xdelta_list = [[1, 1], [10, 10], [10, 10]]; Xabove_lowbounds = [[.5, .1], [4, 7], [1, 6]]

And to sum up: The task was to find prediction values for the 2 sets of features given in the prediction_array. And since our
model do not have training points within the area of these values (the values of the 2 sets of features given in the 
prediction_array are not representet in the training data) we will use the nearest lower and upper points instead. This is 
the points in Xarray_for_interpolation. And for these 8x2 sets of points we have the corresponding 8x2 values in 
Yall_lower_upper.
Now to make the interpolation for the first feature in the 2 sets of features given in the prediction_array (i.e for 3.5 and 
for 2.1) we can substract the first half part of Yall_lower_upper from the last half part. By doing this we find the 
difference in Yall_lower_upper when changing the first feature from lower to upper point - i.e. from 3 to 4 and from 2 to 3 
for the first features in the first and second entry in prediction_array, respectively. 
This gives a Ydelta of:
Ydelta = [[8, 3], [1, 4], [0, 3], [1, 4]], if following the interpolation of the first prediction value we see a difference 
of 8, 1, 0 and 1 when changing (only) the first from 3 to 4. 
We also have that X_delta = [[1, 1], [10, 10], [10, 10]], so now we are able to calc how much Y changes per each change in 
a corresponding feature. For interpolation of first feature X_delta is 1 and 1 for first and second prediction. 
Ydelta/X-delta then gives [[8, 3], [1, 4], [0, 3], [1, 4]]

Now we are not going from 3 to 4, but only from 3 to 3.5. 

Ynext: [[33.  23.3]
[31.5 21.4]
[30.  20.3]
[29.5 23.4]]

modifies now Yall_lower_upper to:

Ynext = 

For each feature we now cut the length of Yall_lower_upper by a factor of 2 by interpolate the first half of Yall_lower_upper with the second half. 
Yall_lower_upper contains values in it's first half that differs from the second half due to the difference in the first feature. 
So if a the first value in row one in Y's first half is 20 (where we have feature one equal to 3) and the corresponding 
first value in row one in Y's second half is 30 (where we have feature one equal to 4) and we have the feature value 
being 3.5 we can now make the first feature interpolation by: Ynext = Y_1st_half + (Ydelta / rowXdelta) * rowXabove. 
And the same for the next feature by cut the length of Yall_lower_upper by a factor of 2 again. When no more features we end up with a single interpolation value
for each individual prediction in prediction_array. E.g [29,31,30]. The values having been interpolated for all features.

"""

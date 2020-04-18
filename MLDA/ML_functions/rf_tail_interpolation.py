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
    """This function is used to decide whether prediction values should be
    sent to simple prediction or interpolation prediction is needed. If 'value'
    is False the variable 'representative' becomes also False, which means that 
    interpolation is needed."""
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
    """Function that collects array with points to be used for interpolatiæon calc.
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


def makeInterpolation(bounds_array=None, prediction_array=None, predictor=None):
    """
    In this function we make the interpolation calculations. First we retrieve the relevant data for the calculations.
    
    Retrieve the relevant data:
    Xdelta_list and Xabove_lowbounds which is found by makeXdeltaListAndXAboveLowbounds. And Xarray_for_interpolation by makeXarrayForInterpolationPredictions.

    Preparing data for interpolation calculations:
    Xarray_for_interpolation contains an array with the shape: (num_predictions * 2**num_feat, num_feat). For each individual prediction in prediction_array 
    (e.g. a individual prediction could be: [3.5, 14, 51]) the Xarray_for_interpolation contains 2**num_feat single row arrays with all combinations of the
    lower/higher points. The 2**num_feat single row arrays for each individual prediction lay on top of each other, the first prediction at the top followed
    by the second and so forth. So if we pass these values: [[3.5, 14, 51], [3.1, 12, 51], [3., 10, 56]] to the prediction_array we would have a 
    Xarray_for_interpolation with 24 rows (num_predictions * 2**num_feat) each row having 3 values. 
    Now passing the values in Xarray_for_interpolation to the predictor then gives num_predictions * 2**num_feat prediction values (Y). 
    Next we reshape and transpose Y so Y now have this shape: (2**num_feat, num_predictions). Which given the ex. above would give a 8 rows and 3 columns shape.

    Interpolation calculations:
    For each feature we now cut the length of Y by a factor of 2 by interpolate the first half of Y with the second half. Y contains values in it's 
    first half that differs from the second half due to the difference in the first feature. So if a the first value in row one in Y's first half is 20 
    (where we have feature one equal to 3) and the corresponding first value in row one in Y's second half is 30 (where we have feature one equal to 4) 
    and we have the feature value being 3.5 we can now make the first feature interpolation by: Ynext = Y_1st_half + (Ydelta / rowXdelta) * rowXabove. 
    And the same for the next feature by cut the length of Y by a factor of 2 again. When no more features we end up with a single interpolation value
    for each individual prediction in prediction_array. E.g [29,31,30]. The values having been interpolated for all features.

    Input: 
    bounds_array: type: list
    prediction_array: type: list or numpy array. Array to make prediction on. 
    predictor: type: see randomForestInterpolation

    Output: type: numpy array. Array with interpolated prediction values.
    """

    # From the bounds_array and prediction_array we find Xdelta_list, Xabove_lowbounds
    # that is the difference in lower and upper point, and how much the prediction value
    # is above the lower point.
    Xdelta_list, Xabove_lowbounds = makeXdeltaListAndXAboveLowbounds(
        bounds_array=bounds_array, prediction_array=prediction_array
    )
    print("Xdelta_list:", Xdelta_list)
    print()
    print("Xabove_lowbounds", Xabove_lowbounds)
    num_predictions, num_features = len(prediction_array), len(prediction_array[0])

    # So far we have the all the closest lower and closest higher points for all
    # our prediction arrays. Now we need to convert all these sets of points (lower/higher)
    # to one long array with a row length of num_predictions*2**num_feat and each row can be
    # calc by the RF_predictor.
    Xarray_for_interpolation = makeXarrayForInterpolationPredictions(
        bounds_array=bounds_array
    )
    print()
    print("Xarray_for_interpolation_shape", np.shape(Xarray_for_interpolation))
    print("Xarray_for_interpolation", Xarray_for_interpolation)
    print()

    Y = f(x=Xarray_for_interpolation, predictor=predictor)
    print("Y:", Y)
    print()
    # Number of columns are equal to num_predictions
    Y = Y.reshape(num_predictions, 2 ** num_features)
    print("Y:", Y)
    print()
    Y = np.transpose(Y)
    print("shape(Y):", np.shape(Y))
    print("Y:", Y)
    print()

    for rowXdelta, rowXabove in zip(Xdelta_list, Xabove_lowbounds):
        Y_1st_half, Y_2nd_half = Y[0 : int(len(Y) / 2)], Y[int(len(Y) / 2) : len(Y)]
        Ydelta = np.array(Y_2nd_half) - np.array(Y_1st_half)
        Ynext = Y_1st_half + (Ydelta / rowXdelta) * rowXabove
        Y = Ynext
    return Y


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
    
    Example:
    Let's see how randomForestInterpolation works by showing an example: say we pass this input (which is a single array) 
    to our prediction_array: [[3.5, 14, 51]] since it has some unrepresentative points. Let's say all points are unrepresentative 
    points. Now to make a prediction we need some representative points to interpolate between. For that we use the 
    collectInterpolationPoints which, let's say give us these values: [[3, 10, 50], [4, 20, 60]] (from the training data). 
    So now we have lower and upper points for all the 3 features in the we want to make prediction for. The next step is to make 
    the interpolation, where we use the makeInterpolation function. makeInterpolation first finds Xdelta_list, Xabove_lowbounds 
    by using makeXdeltaListAndXAboveLowbounds. For this ex Xdelta_list=[[1, 10, 10]] since this is the difference between the 
    lower and upper values shown above, the Xabove_lowbounds = [[.5, 4, 1]] since this is the difference between the lower 
    and the prediction values shown above. 
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
    Now the final step is to make the interpolation between these 8 prediction values. We do that by first looking at the 
    first feature value in our input which is 3.5. Now we do not have any prediction result when the first feature has the value 
    of 3.5. But we have prediction results when the first feature is 3 and 4. The 1st and 5th item in 'combinations' above is 
    [3, 10, 50] and [4, 10, 50] and the corresponding prediction results are 29 and 37. Note the two other features are the 
    same in 1st and 5th item (i.e. 10 and 50, respectively). So the difference in prediction results (the difference between 
    29 and 37) is then only due to the difference in the first feature (i.e. between 3 and 4, respectively). 
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

    Input: 
    prediction_array: type: list or numpy array. Array to make prediction on. 
    xdata_names: type: pandas.core.indexes.base.Index
    names_bounds: type: dict
    xdata: type: pandas dataframe
    RF_predictor: type: sklearn.ensemble.forest.RandomForestRegressor
    criteria_method, min_points, min_percentage, area_frac_to_search_points: see randomForestPredictorWithInterpolation

    Output: type: numpy array. Array with interpolated prediction values.
    """
    print("type(xdata_names", type(RF_predictor))
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
    # Now interpolation calc
    Y = makeInterpolation(
        bounds_array=bounds_array,
        prediction_array=prediction_array,
        predictor=RF_predictor,
    )
    return Y[0]


def randomForestPredictorWithInterpolation(
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


# Input
ML_dictEN2 = load_object("/home/jesper/Work/MLDA_app/MLDA/jupyter_ML/ML_dictEN2.sav")
values = [[416.5, 125.5, 4.0, 0.0, 0.0], [316.5, 134.75, 4.0, 0.0, 0.0]]
predict_y = randomForestPredictorWithInterpolation(
    prediction_array=values, ML_dict=ML_dictEN2, yvar="heat_load"
)
print(predict_y)
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


# prediction_values = [
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
#     [350, 161, 2.5, 0.25, 4.5],
#     [250, 131, 3.5, 0.35, 0.5],
#     [245.5, 210, 4.5, 0.15, 2.5],
# ]

# prediction_values2 = [
#     [318.5, 110.25, 2.0, 0.234375, 2.8125],
#     [318.5, 122.5, 2.0, 0.234375, 2.8125],
#     [318.5, 134.75, 2.0, 0.234375, 2.8125],
#     [318.5, 147.0, 2.0, 0.234375, 2.8125],
#     [318.5, 159.25, 2.0, 0.234375, 2.8125],
#     [318.5, 171.5, 2.0, 0.234375, 2.8125],
#     [318.5, 183.75, 2.0, 0.234375, 2.8125],
#     [318.5, 196.0, 2.0, 0.234375, 2.8125],
#     [318.5, 208.25, 2.0, 0.234375, 2.8125],
#     [318.5, 220.5, 2.0, 0.234375, 2.8125],
#     [318.5, 110.25, 2.2142857142857144, 0.234375, 2.8125],
#     [318.5, 122.5, 2.2142857142857144, 0.234375, 2.8125],
#     [318.5, 134.75, 2.2142857142857144, 0.234375, 2.8125],
#     [318.5, 147.0, 2.2142857142857144, 0.234375, 2.8125],
#     [318.5, 159.25, 2.2142857142857144, 0.234375, 2.8125],
#     [318.5, 171.5, 2.2142857142857144, 0.234375, 2.8125],
#     [318.5, 183.75, 2.2142857142857144, 0.234375, 2.8125],
#     [318.5, 196.0, 2.2142857142857144, 0.234375, 2.8125],
#     [318.5, 208.25, 2.2142857142857144, 0.234375, 2.8125],
#     [318.5, 220.5, 2.2142857142857144, 0.234375, 2.8125],
#     [318.5, 110.25, 2.4285714285714284, 0.234375, 2.8125],
#     [318.5, 122.5, 2.4285714285714284, 0.234375, 2.8125],
#     [318.5, 134.75, 2.4285714285714284, 0.234375, 2.8125],
#     [318.5, 147.0, 2.4285714285714284, 0.234375, 2.8125],
#     [318.5, 159.25, 2.4285714285714284, 0.234375, 2.8125],
#     [318.5, 171.5, 2.4285714285714284, 0.234375, 2.8125],
#     [318.5, 183.75, 2.4285714285714284, 0.234375, 2.8125],
#     [318.5, 196.0, 2.4285714285714284, 0.234375, 2.8125],
#     [318.5, 208.25, 2.4285714285714284, 0.234375, 2.8125],
#     [318.5, 220.5, 2.4285714285714284, 0.234375, 2.8125],
#     [318.5, 110.25, 2.642857142857143, 0.234375, 2.8125],
#     [318.5, 122.5, 2.642857142857143, 0.234375, 2.8125],
#     [318.5, 134.75, 2.642857142857143, 0.234375, 2.8125],
#     [318.5, 147.0, 2.642857142857143, 0.234375, 2.8125],
#     [318.5, 159.25, 2.642857142857143, 0.234375, 2.8125],
#     [318.5, 171.5, 2.642857142857143, 0.234375, 2.8125],
#     [318.5, 183.75, 2.642857142857143, 0.234375, 2.8125],
#     [318.5, 196.0, 2.642857142857143, 0.234375, 2.8125],
#     [318.5, 208.25, 2.642857142857143, 0.234375, 2.8125],
#     [318.5, 220.5, 2.642857142857143, 0.234375, 2.8125],
#     [318.5, 110.25, 2.857142857142857, 0.234375, 2.8125],
#     [318.5, 122.5, 2.857142857142857, 0.234375, 2.8125],
#     [318.5, 134.75, 2.857142857142857, 0.234375, 2.8125],
#     [318.5, 147.0, 2.857142857142857, 0.234375, 2.8125],
#     [318.5, 159.25, 2.857142857142857, 0.234375, 2.8125],
#     [318.5, 171.5, 2.857142857142857, 0.234375, 2.8125],
#     [318.5, 183.75, 2.857142857142857, 0.234375, 2.8125],
#     [318.5, 196.0, 2.857142857142857, 0.234375, 2.8125],
#     [318.5, 208.25, 2.857142857142857, 0.234375, 2.8125],
#     [318.5, 220.5, 2.857142857142857, 0.234375, 2.8125],
#     [318.5, 110.25, 3.071428571428571, 0.234375, 2.8125],
#     [318.5, 122.5, 3.071428571428571, 0.234375, 2.8125],
#     [318.5, 134.75, 3.071428571428571, 0.234375, 2.8125],
#     [318.5, 147.0, 3.071428571428571, 0.234375, 2.8125],
#     [318.5, 159.25, 3.071428571428571, 0.234375, 2.8125],
#     [318.5, 171.5, 3.071428571428571, 0.234375, 2.8125],
#     [318.5, 183.75, 3.071428571428571, 0.234375, 2.8125],
#     [318.5, 196.0, 3.071428571428571, 0.234375, 2.8125],
#     [318.5, 208.25, 3.071428571428571, 0.234375, 2.8125],
#     [318.5, 220.5, 3.071428571428571, 0.234375, 2.8125],
#     [318.5, 110.25, 3.2857142857142856, 0.234375, 2.8125],
#     [318.5, 122.5, 3.2857142857142856, 0.234375, 2.8125],
#     [318.5, 134.75, 3.2857142857142856, 0.234375, 2.8125],
#     [318.5, 147.0, 3.2857142857142856, 0.234375, 2.8125],
#     [318.5, 159.25, 3.2857142857142856, 0.234375, 2.8125],
#     [318.5, 171.5, 3.2857142857142856, 0.234375, 2.8125],
#     [318.5, 183.75, 3.2857142857142856, 0.234375, 2.8125],
#     [318.5, 196.0, 3.2857142857142856, 0.234375, 2.8125],
#     [318.5, 208.25, 3.2857142857142856, 0.234375, 2.8125],
#     [318.5, 220.5, 3.2857142857142856, 0.234375, 2.8125],
#     [318.5, 110.25, 3.5, 0.234375, 2.8125],
#     [318.5, 122.5, 3.5, 0.234375, 2.8125],
#     [318.5, 134.75, 3.5, 0.234375, 2.8125],
#     [318.5, 147.0, 3.5, 0.234375, 2.8125],
#     [318.5, 159.25, 3.5, 0.234375, 2.8125],
#     [318.5, 171.5, 3.5, 0.234375, 2.8125],
#     [318.5, 183.75, 3.5, 0.234375, 2.8125],
#     [318.5, 196.0, 3.5, 0.234375, 2.8125],
#     [318.5, 208.25, 3.5, 0.234375, 2.8125],
#     [318.5, 220.5, 3.5, 0.234375, 2.8125],
#     [318.5, 110.25, 3.7142857142857144, 0.234375, 2.8125],
#     [318.5, 122.5, 3.7142857142857144, 0.234375, 2.8125],
#     [318.5, 134.75, 3.7142857142857144, 0.234375, 2.8125],
#     [318.5, 147.0, 3.7142857142857144, 0.234375, 2.8125],
#     [318.5, 159.25, 3.7142857142857144, 0.234375, 2.8125],
#     [318.5, 171.5, 3.7142857142857144, 0.234375, 2.8125],
#     [318.5, 183.75, 3.7142857142857144, 0.234375, 2.8125],
#     [318.5, 196.0, 3.7142857142857144, 0.234375, 2.8125],
#     [318.5, 208.25, 3.7142857142857144, 0.234375, 2.8125],
#     [318.5, 220.5, 3.7142857142857144, 0.234375, 2.8125],
#     [318.5, 110.25, 3.9285714285714284, 0.234375, 2.8125],
#     [318.5, 122.5, 3.9285714285714284, 0.234375, 2.8125],
#     [318.5, 134.75, 3.9285714285714284, 0.234375, 2.8125],
#     [318.5, 147.0, 3.9285714285714284, 0.234375, 2.8125],
#     [318.5, 159.25, 3.9285714285714284, 0.234375, 2.8125],
#     [318.5, 171.5, 3.9285714285714284, 0.234375, 2.8125],
#     [318.5, 183.75, 3.9285714285714284, 0.234375, 2.8125],
#     [318.5, 196.0, 3.9285714285714284, 0.234375, 2.8125],
#     [318.5, 208.25, 3.9285714285714284, 0.234375, 2.8125],
#     [318.5, 220.5, 3.9285714285714284, 0.234375, 2.8125],
#     [318.5, 110.25, 4.142857142857142, 0.234375, 2.8125],
#     [318.5, 122.5, 4.142857142857142, 0.234375, 2.8125],
#     [318.5, 134.75, 4.142857142857142, 0.234375, 2.8125],
#     [318.5, 147.0, 4.142857142857142, 0.234375, 2.8125],
#     [318.5, 159.25, 4.142857142857142, 0.234375, 2.8125],
#     [318.5, 171.5, 4.142857142857142, 0.234375, 2.8125],
#     [318.5, 183.75, 4.142857142857142, 0.234375, 2.8125],
#     [318.5, 196.0, 4.142857142857142, 0.234375, 2.8125],
#     [318.5, 208.25, 4.142857142857142, 0.234375, 2.8125],
#     [318.5, 220.5, 4.142857142857142, 0.234375, 2.8125],
#     [318.5, 110.25, 4.357142857142858, 0.234375, 2.8125],
#     [318.5, 122.5, 4.357142857142858, 0.234375, 2.8125],
#     [318.5, 134.75, 4.357142857142858, 0.234375, 2.8125],
#     [318.5, 147.0, 4.357142857142858, 0.234375, 2.8125],
#     [318.5, 159.25, 4.357142857142858, 0.234375, 2.8125],
#     [318.5, 171.5, 4.357142857142858, 0.234375, 2.8125],
#     [318.5, 183.75, 4.357142857142858, 0.234375, 2.8125],
#     [318.5, 196.0, 4.357142857142858, 0.234375, 2.8125],
#     [318.5, 208.25, 4.357142857142858, 0.234375, 2.8125],
#     [318.5, 220.5, 4.357142857142858, 0.234375, 2.8125],
#     [318.5, 110.25, 4.571428571428571, 0.234375, 2.8125],
#     [318.5, 122.5, 4.571428571428571, 0.234375, 2.8125],
#     [318.5, 134.75, 4.571428571428571, 0.234375, 2.8125],
#     [318.5, 147.0, 4.571428571428571, 0.234375, 2.8125],
#     [318.5, 159.25, 4.571428571428571, 0.234375, 2.8125],
#     [318.5, 171.5, 4.571428571428571, 0.234375, 2.8125],
#     [318.5, 183.75, 4.571428571428571, 0.234375, 2.8125],
#     [318.5, 196.0, 4.571428571428571, 0.234375, 2.8125],
#     [318.5, 208.25, 4.571428571428571, 0.234375, 2.8125],
#     [318.5, 220.5, 4.571428571428571, 0.234375, 2.8125],
#     [318.5, 110.25, 4.785714285714286, 0.234375, 2.8125],
#     [318.5, 122.5, 4.785714285714286, 0.234375, 2.8125],
#     [318.5, 134.75, 4.785714285714286, 0.234375, 2.8125],
#     [318.5, 147.0, 4.785714285714286, 0.234375, 2.8125],
#     [318.5, 159.25, 4.785714285714286, 0.234375, 2.8125],
#     [318.5, 171.5, 4.785714285714286, 0.234375, 2.8125],
#     [318.5, 183.75, 4.785714285714286, 0.234375, 2.8125],
#     [318.5, 196.0, 4.785714285714286, 0.234375, 2.8125],
#     [318.5, 208.25, 4.785714285714286, 0.234375, 2.8125],
#     [318.5, 220.5, 4.785714285714286, 0.234375, 2.8125],
#     [318.5, 110.25, 5.0, 0.234375, 2.8125],
#     [318.5, 122.5, 5.0, 0.234375, 2.8125],
#     [318.5, 134.75, 5.0, 0.234375, 2.8125],
#     [318.5, 147.0, 5.0, 0.234375, 2.8125],
#     [318.5, 159.25, 5.0, 0.234375, 2.8125],
#     [318.5, 171.5, 5.0, 0.234375, 2.8125],
#     [318.5, 183.75, 5.0, 0.234375, 2.8125],
#     [318.5, 196.0, 5.0, 0.234375, 2.8125],
#     [318.5, 208.25, 5.0, 0.234375, 2.8125],
#     [318.5, 220.5, 5.0, 0.234375, 2.8125],
# ]


# Y = randomForestPredictorWithInterpolation(
#     prediction_array=prediction_values, ML_dict=ML_dictEN2, yvar="heat_load"
# )
# print(Y)

# Y = randomForestPredictorWithInterpolation(
#     prediction_array=prediction_values2, ML_dict=ML_dictEN2, yvar="heat_load"
# )
# print(Y)


# Y = f(x=prediction_values, predictor=rf_predictor)
# print(Y)


# Y = makeInterpolation(bounds_array=bounds_array, prediction_array=prediction_values)
# print(Y)

# # Input
# prediction_values = [[3.5, 59, 22, 34], [3.2, 55, 21, 38], [3.5, 59, 22, 34]]

# bounds_array = [
#     [[3, 4], [50, 60], [20, 30], [30, 40]],
#     [[3, 4], [50, 60], [20, 30], [30, 40]],
#     [[3, 4], [50, 60], [20, 30], [30, 40]],
# ]

# sum_sum_list = makeXarrayForInterpolationPredictions(bounds_array=bounds_array)
# a = np.array(
#     (
#         [
#             2,
#             2.5,
#             2.25,
#             1.7,
#             1.3,
#             1,
#             0.45,
#             0.75,
#             0.7,
#             3.25,
#             2.8,
#             2.4,
#             2.2,
#             1.9,
#             1.5,
#             1.4,
#         ],
#         [
#             21,
#             2.9,
#             2.25,
#             2.7,
#             1.3,
#             1,
#             0.45,
#             0.75,
#             1.7,
#             3.25,
#             2.9,
#             2.4,
#             2.2,
#             1.9,
#             2.5,
#             2.4,
#         ],
#         [
#             2,
#             2.5,
#             2.25,
#             1.7,
#             1.3,
#             1,
#             0.45,
#             0.75,
#             0.7,
#             3.25,
#             2.8,
#             2.4,
#             2.2,
#             1.9,
#             1.5,
#             1.4,
#         ],
#     )
# )


# print(len(sum_sum_list))
# print(len(a[0]))

# Y = makeInterpolation(
#     Y=a, bounds_array=bounds_array, prediction_array=prediction_values
# )
# print(Y)


# def keepIfDataPointsAreComprisingPercentageAboveLimit(
#     data_fraction_limit=None,
#     list_of_values=None,
#     percentage_of_range=None,
#     xdata=None,
#     var=None,
# ):
#     """Function that takes list of values and returns another list of values which passed the conditions. The overall
#     approach is to ensure that we base our interpolation on a representatative amount of data. In other words, if we
#     want to make a prediction and one of our variables is not representet in our train dataset, i.e. not close to any
#     of the train data that our model is based on, then we want to make a interpolation from data that our model is based on.
#     And in order to say that these data are representatative is to say that they comprise a certain fraction of the total
#     amount of data (data_fraction_limit), and that we collect the data for interpolation in a certain confined area (percentage_of_range).
#     Approach: the conditions to pass is the data_fraction_limit and percentage_of_range. The data_fraction_limit is
#     the minimum fraction of data (of the total dataset) we want to base the points of interpolation on. The percentage_of_range
#     is the distance range for the given variable, i.e. how far to the left and right we can go, when we collect data to calc
#     the data_fraction_limit. When data_fraction_limit is higher and the percentage_of_range is lower, the conditions are more difficult
#     to fullfill, since then we both want our data to represent a higher fraction of the total amount of data, while we also want to find these data on
#     a small area (not far to the left and right).
#     Input:
#     data_fraction_limit:fraction we specify,
#     list_of_values: values that are either rejected or returned by the function,
#     percentage_of_range: fraction we specify,
#     xdata: dataframe,
#     var: variable
#     Output:
#     list_result: values that passed the conditions.
#     """
#     list_result = []
#     for item in list_of_values:
#         low, high = item - percentage_of_range, item + percentage_of_range
#         if (
#             xdata[var][(xdata[var] > low) & (xdata[var] < high)].count()
#             / xdata[var].count()
#             > data_fraction_limit
#         ):
#             list_result.append(item)
#     return list_result


# def findBelowAndAboveValuesUsingRangeAndLimit(
#     xdata=None,
#     var=None,
#     to_predict=None,
#     percent_value_data_range=None,
#     data_fraction_limit=None,
# ):
#     list_lower = xdata[var][xdata[var] <= to_predict[var]].drop_duplicates()
#     list_higher = xdata[var][xdata[var] >= to_predict[var]].drop_duplicates()
#     # find data range for var
#     lowest, highest = xdata[var].min(), xdata[var].max()
#     l_h_range = highest - lowest  # lowest, highest range
#     percentage_of_range = l_h_range * percent_value_data_range

#     # llksadfl: list_lower_keep_since_above_data_fraction_limit
#     llksadfl = keepIfDataPointsAreComprisingPercentageAboveLimit(
#         data_fraction_limit=data_fraction_limit,
#         list_of_values=list_lower,
#         percentage_of_range=percentage_of_range,
#         xdata=xdata,
#         var=var,
#     )
#     # lhksadfl: list_higher_keep_since_above_data_fraction_limit
#     lhksadfl = keepIfDataPointsAreComprisingPercentageAboveLimit(
#         data_fraction_limit=0.05,
#         list_of_values=list_higher,
#         percentage_of_range=percentage_of_range,
#         xdata=xdata,
#         var=var,
#     )
#     return llksadfl, lhksadfl


# def collectValuesForLinRegr(
#     df_xdata=None,
#     to_predict=None,
#     threshold=0.1,
#     percent_value_data_range=0.05,
#     data_fraction_limit=0.02,
# ):
#     """Approach: If prediction value (var) is not outside the threshold, keep the value (i.e. append it two times).
#     When the prediction value (var) is not outside the threshold it means it is close to some of the models training data points.
#     And this means again that no interpolation is needed. Else we will find interpolation points by finding points below and above
#     the prediction value."""
#     xdata = df_xdata
#     to_predict_for_regr = {key: [] for key in xdata.columns}
#     for var in to_predict:
#         # First we find the closest lower/higher value to the prediction value (var),
#         # NOTE: the closest value could be the prediction value.
#         closest_lower = xdata[var][xdata[var] <= to_predict[var]].max()
#         closest_higher = xdata[var][xdata[var] >= to_predict[var]].min()
#         # Then we check if it inside the threshold - i.e. is either closest_lower or closest_higher sufficient close
#         # to the prediction value. And keep the value if it is (i.e. append it two times).
#         if closest_lower > to_predict[var] * (
#             1 - threshold
#         ) or closest_higher < to_predict[var] * (1 + threshold):
#             for i in range(2):
#                 to_predict_for_regr[var].append(to_predict[var])
#         else:
#             # Arriving here in the algoritme means that no training data are
#             # close to the value for prediction for this var, and thus we will
#             # find interpolation points by finding points below and above
#             # the prediction value.
#             # llksadfl: list_lower_keep_since_above_data_fraction_limit
#             # lhksadfl: list_higher_keep_since_above_data_fraction_limit
#             llksadfl, lhksadfl = findBelowAndAboveValuesUsingRangeAndLimit(
#                 xdata=xdata,
#                 var=var,
#                 to_predict=to_predict,
#                 percent_value_data_range=percent_value_data_range,
#                 data_fraction_limit=data_fraction_limit,
#             )
#             if len(llksadfl) == 0 or len(lhksadfl) == 0:
#                 return f"Unable to find interpolation data points given the input: percent_value_data_range={percent_value_data_range} and data_fraction_limit={data_fraction_limit}"
#             else:
#                 # Now finding the value closest to the prediction value
#                 closest_lower = max(llksadfl)
#                 closest_higher = min(lhksadfl)
#             # Now appending the data points below and above the prediction value
#             to_predict_for_regr[var].append(closest_lower)
#             to_predict_for_regr[var].append(closest_higher)
#     return to_predict_for_regr


# def makeDFforMultipleRegr(to_predict=None, df_xdata=None, threshold=0.1):
#     """Function that makes a dataframe for multiple interpolation of vars.
#     Approach: The function creates data for multiple linear regression by making
#     rows starting with 2 lower values for var. For the first low value of
#     var we set all the other vars to their lower value in that row. For the
#     next low value of var we set all the other vars to their higher value.
#     And repeating that for but with the 2 high values for var. And then the
#     same for the next var.
#     Input:
#     to_predict: dict with the values to predict Y,
#     df_xdata: a dataframe with the train xdata
#     threshold: threshold to pass to the function collectValuesForLinRegr
#     """
#     to_predict_for_regr = collectValuesForLinRegr(
#         to_predict=to_predict, df_xdata=df_xdata, threshold=threshold
#     )
#     collect_to_df = {key: [] for key in to_predict_for_regr}
#     count_vars_no_regr = 0
#     for var in to_predict_for_regr:
#         # If var in to_predict_for_regr have different values
#         if to_predict_for_regr[var][0] != to_predict_for_regr[var][1]:
#             # then we append these two values 2 times
#             for i in range(2):
#                 collect_to_df[var].append(to_predict_for_regr[var][0])
#             for i in range(2):
#                 collect_to_df[var].append(to_predict_for_regr[var][1])
#             # then we fill out the other
#             for name in to_predict_for_regr:
#                 if name == var:
#                     continue
#                 for i in range(2):
#                     collect_to_df[name].append(to_predict_for_regr[name][0])
#                     collect_to_df[name].append(to_predict_for_regr[name][1])
#         else:
#             count_vars_no_regr += 1
#     # if all vars are below the threshold - i.e. no interpolation
#     if count_vars_no_regr == len(to_predict_for_regr):
#         return "All vars are below the threshold - i.e. no interpolation"
#     df_regr = pd.DataFrame(data=collect_to_df)
#     df_regr = df_regr.drop_duplicates()
#     return df_regr, to_predict_for_regr


# def findSlopes(df_xdata_regr=None, predictor=None):
#     """Find slopes for all the relevant vars. Using the adaptable function f."""
#     # we copy df_xdata, since now we add Y to the dataframe
#     df_regr = df_xdata_regr.copy()
#     df_regr["Y"] = df_regr[df_xdata_regr.columns].apply(f, args=(predictor,), axis=1)
#     X = df_regr[df_xdata_regr.columns].to_numpy()
#     Y = df_regr[["Y"]].to_numpy()
#     model = LinearRegression().fit(X, Y)
#     return model.coef_, df_regr


# def intepolate(df_regr=None, slope=None, to_predict=None, to_predict_for_regr=None):
#     """Based on slopes and data, multiple var interpolation is made.
#     Approach: we start out with the Y-value when all X-values are lowest (interpol_Y1), then we add
#     change in Y (delta_Y) due to change in X (interpol_X - interpol_X1).
#     The function returns the prediction of Y.
#     Input:
#     df_regr: dataframe with combinations of nearest lowest and nearest highest from to_predict_for_regr, incl. Y predictions.
#     slope: array with slopes
#     to_predict: dict with the values to predict Y
#     to_predict_for_regr: dict with nearest lowest and nearest highest or - if predict value for var is below threshold - predict value twice.
#     """
#     # The first row in df is when all X-values are lowest, which we call interpol_Y1
#     interpol_Y1 = df_regr.loc[0, ["Y"]].to_numpy()
#     interpol_X1 = np.array([to_predict_for_regr[num][0] for num in to_predict_for_regr])
#     # NOTE: To calc delta_Y we have to make sure that the vars in interpol_X1 and interpol_X
#     # comes in the same order. To ensure that we iterate through 'to_predict_for_regr' in the
#     # list comprehension for both interpol_X1 and interpol_X.
#     interpol_X = np.array([to_predict[num] for num in to_predict_for_regr])
#     delta_Y = np.sum(slope * (interpol_X - interpol_X1))
#     return interpol_Y1 + delta_Y


# def refineRF_PredictionByInterpolation(
#     xdata_names=None,
#     xdata_numpy=None,
#     to_predict_dict=None,
#     threshold=0.1,
#     predictor=None,
# ):
#     """Coordinator function. Function that predicts Y by calling the relevant functions.
#     Input:
#     to_predict: dict with the values to predict Y,
#     df_xdata: a dataframe with the train xdata
#     threshold: threshold to pass to the function collectValuesForLinRegr
#     """
#     df_xdata = pd.DataFrame(data=xdata_numpy, columns=xdata_names)
#     # First we check if all the vars are below the threshold
#     if (
#         makeDFforMultipleRegr(
#             to_predict=to_predict_dict, df_xdata=df_xdata, threshold=threshold
#         )
#         == "All vars are below the threshold - i.e. no interpolation"
#     ):
#         return "All vars are below the threshold - i.e. no interpolation"

#     df_xdata_regr, to_predict_for_regr = makeDFforMultipleRegr(
#         to_predict=to_predict_dict, df_xdata=df_xdata, threshold=threshold
#     )
#     slope, df_regr = findSlopes(df_xdata_regr=df_xdata_regr, predictor=predictor)
#     predict_Y = intepolate(
#         df_regr=df_regr,
#         slope=slope,
#         to_predict=to_predict_dict,
#         to_predict_for_regr=to_predict_for_regr,
#     )
#     return predict_Y[0][0]


# """Not Used"""


# def findSlopeSingleVar(df_xdata_regr=None, predictor=None, var=None):
#     """Find slope for when one var has two different values, all other vars have same values.
#      Using the adaptable function f."""
#     # we copy df_xdata, since now we add Y to the dataframe
#     df_regr = df_xdata_regr.copy()
#     # Calc Y by applying predictor to x array
#     df_regr["Y"] = df_regr[df_xdata_regr.columns].apply(f, args=(predictor,), axis=1)
#     # Since we only find the slope for a single value of the x values we
#     # shrink the df to only include this x
#     df_regr["X"] = df_regr[var]
#     # Convert to numpy
#     X = df_regr["X"].to_numpy().reshape(-1, 1)
#     Y = df_regr[["Y"]].to_numpy()
#     # Make regression
#     model = LinearRegression().fit(X, Y)
#     return model.coef_  # return slope of single xvar


# # # Exercise
# # dict_test = load_object(
# #     "/home/jesper/Work/MLDA_app/MLDA/input_data/train_test_dict.sav"
# # )
# # xdata_names = dict_test["N1"]["xdata_names"]
# # xdata_numpy = dict_test["N1"]["X"]["X_train"]
# # to_predict = {
# #     "glaz_area_distrib": 0.0,
# #     "glazing_area": 0.0,
# #     "height": 4.5,
# #     "roof_area": 220.5,
# #     "surf_area": 686.0,
# #     "wall_area": 245.0,
# # }
# # predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]

# # actual = refineRF_PredictionByInterpolation(
# #     xdata_names=xdata_names,
# #     xdata_numpy=xdata_numpy,
# #     to_predict_dict=to_predict,
# #     threshold=0.1,
# #     predictor=predictor,
# # )

# # print(actual)


# # dict_test = load_object(
# #     "/home/jesper/Work/MLDA_app/MLDA/input_data/train_test_dict.sav"
# # )
# # predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]
# # xdata_names = dict_test["N1"]["xdata_names"]
# # xdata_numpy = dict_test["N1"]["X"]["X_train"]

# # # Exercise
# # dict_test = load_object(
# #     "/home/jesper/Work/MLDA_app/MLDA/input_data/train_test_dict.sav"
# # )
# # to_predict = {
# #     "glaz_area_distrib": 0,
# #     "glazing_area": 0,
# #     "height": 3.5,
# #     "roof_area": 220,
# #     "surf_area": 686,
# #     "wall_area": 245,
# # }
# # xdata_names = dict_test["N1"]["xdata_names"]
# # xdata_numpy = dict_test["N1"]["X"]["X_train"]
# # df_xdata = pd.DataFrame(data=xdata_numpy, columns=xdata_names)
# # print(df_xdata["glazing_area"].value_counts())


# # hey = keepIfDataPointsAreComprisingPercentageAboveLimit(
# #     data_fraction_limit=0.1,
# #     list_of_values=[0.00, 0.10, 0.25, 0.40],
# #     percentage_of_range=0.05,
# #     xdata=df_xdata,
# #     var="glazing_area",
# # )

# # print(hey)
# # actual = makeDFforMultipleRegr(to_predict=to_predict, df_xdata=df_xdata, threshold=0.1)

# # print(actual)
# # to_predict = {
# #     "glaz_area_distrib": 0.0,
# #     "glazing_area": 0.0,
# #     "height": 3.5,
# #     "roof_area": 187.0,
# #     "surf_area": 686,
# #     "wall_area": 400.0,
# # }
# # xdata_names = dict_test["N1"]["xdata_names"]
# # xdata_numpy = dict_test["N1"]["X"]["X_train"]
# # df_xdata = pd.DataFrame(data=xdata_numpy, columns=xdata_names)
# # predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]


# # test = collectValuesForLinRegr(
# #     df_xdata=df_xdata,
# #     to_predict=to_predict,
# #     threshold=0.08,
# #     percent_value_data_range=0.05,
# #     data_fraction_limit=0.03,
# # )

# # predict = refineRF_PredictionByInterpolation(
# #     xdata_names=xdata_names,
# #     xdata_numpy=xdata_numpy,
# #     predictor=predictor,
# #     to_predict_dict=to_predict,
# #     threshold=0.1,
# # )


# # print(to_predict)
# # print(test)
# # print(predict)


# # df = pd.DataFrame(data=xdata_numpy, columns=xdata_names)
# # df_one_row = df.loc[38, :]
# # print(df_one_row.shape)

# # df2 = pd.DataFrame(data=[[686, 245, 220.5, 3.5, 0, 0]], columns=xdata_names)
# # df_one_row2 = df2.loc[0, :]
# # print(df_one_row2.shape)


# # x = np.array([686, 245, 220.5, 3.5, 0, 0])


# # print(f(x=x, predictor=predictor))
# # print(f(x=df_one_row, predictor=predictor))
# # # print(df_one_row)

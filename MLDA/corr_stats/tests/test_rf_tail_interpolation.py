import numpy as np
import pandas as pd
import pytest
import numpy.testing as npt
from pandas.testing import assert_frame_equal
import joblib

from MLDA.miscellaneous.save_and_load_files import load_object, save_object
from MLDA.ML_functions.rf_tail_interpolation import (
    f,
    keepIfDataPointsAreComprisingPercentageAboveLimit,
    findBelowAndAboveValuesUsingRangeAndLimit,
    collectValuesForLinRegr,
    makeDFforMultipleRegr,
    findSlopes,
    intepolate,
    refineRF_PredictionByInterpolation,
)


"""Dataset used for test"""


@pytest.fixture()
def dictForTest():
    dict_test = load_object(
        "/home/jesper/Work/MLDA_app/MLDA/input_data/train_test_dict.sav"
    )
    return dict_test


"""END Dataset used for test"""


"""Testing the function f"""


def test_f_when_input_is_list_with_one_entry(dictForTest):
    # Setup
    desired = 6.048

    # Exercise
    dict_test = dictForTest
    predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]
    xdata_names = dict_test["N1"]["xdata_names"]
    dict1 = {
        "glaz_area_distrib": [0.0, 0.0],
        "glazing_area": [0.0, 0.0],
        "height": [3.5, 3.5],
        "roof_area": [147.0, 220.5],
        "surf_area": [661.5, 686.0],
        "wall_area": [245.0, 245.0],
    }
    dict1_to_list = [
        dict1[key][1] for key in xdata_names
    ]  # takes the second value of all var in dict1

    actual = f(x=dict1_to_list, predictor=predictor)

    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)  #


def test_f_when_input_is_list_with_two_entries(dictForTest):
    # Setup
    desired = [6.048, 6.048]

    # Exercise
    dict_test = dictForTest
    predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]
    xdata_names = dict_test["N1"]["xdata_names"]
    dict1 = {
        "glaz_area_distrib": [0.0, 0.0],
        "glazing_area": [0.0, 0.0],
        "height": [3.5, 3.5],
        "roof_area": [147.0, 220.5],
        "surf_area": [661.5, 686.0],
        "wall_area": [245.0, 245.0],
    }
    dict1_to_list = [
        [dict1[key][1] for key in xdata_names],
        [dict1[key][1] for key in xdata_names],
    ]  # takes the second value of all var in dict1, two times

    actual = f(x=dict1_to_list, predictor=predictor)

    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)  #


def test_f_when_input_is_a_numpy_array_with_one_entry(dictForTest):
    # Setup
    desired = 6.048

    # Exercise
    dict_test = dictForTest
    predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]
    x = np.array([686, 245, 220.5, 3.5, 0, 0])

    actual = f(x=x, predictor=predictor)

    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)  #


def test_f_when_input_is_a_pandas_series_with_one_row_entry(dictForTest):
    # Setup
    desired = 6.048

    # Exercise
    dict_test = dictForTest
    xdata_names = dict_test["N1"]["xdata_names"]
    predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]
    df = pd.DataFrame(data=[[686, 245, 220.5, 3.5, 0, 0]], columns=xdata_names)
    # df_one_row = df.loc[0, :]

    actual = f(x=df, predictor=predictor)

    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)  #


def test_f_when_input_is_a_pandas_series_with_two_rows(dictForTest):
    # Setup
    desired = [6.048, 16.8034]

    # Exercise
    dict_test = dictForTest
    predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]
    xdata_names = dict_test["N1"]["xdata_names"]
    df = pd.DataFrame(
        data=[[686, 245, 220.5, 3.5, 0, 0], [696, 145, 120.5, 6.5, 0, 0]],
        columns=xdata_names,
    )
    # df_one_row = df.loc[0, :]

    actual = f(x=df, predictor=predictor)

    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)  #


"""END Testing the function f"""

"""Testing the function keepIfDataPointsAreComprisingPercentageAboveLimit"""


def test_keepIfDataPointsAreComprisingPercentageAboveLimit_returns_a_list_with_items_but_with_one_item_remove_from_input_list_since_it_is_below_the_data_fraction_limit(
    dictForTest
):
    # Setup
    desired = [0.1, 0.25, 0.4]

    # Exercise
    dict_test = dictForTest
    xdata_names = dict_test["N1"]["xdata_names"]
    xdata_numpy = dict_test["N1"]["X"]["X_train"]
    df_xdata = pd.DataFrame(data=xdata_numpy, columns=xdata_names)

    actual = keepIfDataPointsAreComprisingPercentageAboveLimit(
        data_fraction_limit=0.1,
        list_of_values=[0.00, 0.10, 0.25, 0.40],
        percentage_of_range=0.05,
        xdata=df_xdata,
        var="glazing_area",
    )

    # Verify
    assert actual == desired


def test_keepIfDataPointsAreComprisingPercentageAboveLimit_returns_only_numbers_that_passes_the__percentage_of_range__condition():
    # Setup
    desired = [3]

    # Exercise
    xdata_names = ["numbers_around_3", "zeroes"]
    np.random.seed(1)
    numbers_around_3 = np.random.normal(loc=3, size=30, scale=0.5)
    zeroes = np.zeros_like(numbers_around_3)
    xdata_numpy = [numbers_around_3, zeroes]
    xdata_as_dict = {key: np_list for key, np_list in zip(xdata_names, xdata_numpy)}
    df_xdata = pd.DataFrame(data=xdata_as_dict)

    actual = keepIfDataPointsAreComprisingPercentageAboveLimit(
        data_fraction_limit=0.1,
        list_of_values=[0, 1, 2, 3, 4],
        percentage_of_range=0.15,
        xdata=df_xdata,
        var="numbers_around_3",
    )

    # Verify
    assert actual == desired


"""END Testing the function keepIfDataPointsAreComprisingPercentageAboveLimit"""


"""Testing the function collectValuesForLinRegr"""


def test_collectValuesForLinRegr_gets_input_values_below_threshold_so_should_return_value_twice_for_each_var(
    dictForTest
):
    # Setup
    desired = {
        "glaz_area_distrib": [0.0, 0.0],
        "glazing_area": [0.0, 0.0],
        "height": [3.5, 3.5],
        "roof_area": [220.5, 220.5],
        "surf_area": [686.0, 686.0],
        "wall_area": [245.0, 245.0],
    }

    # Exercise
    dict_test = dictForTest
    dict1 = {
        "glaz_area_distrib": [0.0, 0.0],
        "glazing_area": [0.0, 0.0],
        "height": [3.5, 3.5],
        "roof_area": [220.5, 220.5],
        "surf_area": [686.0, 686.0],
        "wall_area": [245.0, 245.0],
    }
    xdata_names = dict_test["N1"]["xdata_names"]
    xdata_numpy = dict_test["N1"]["X"]["X_train"]
    df_xdata = pd.DataFrame(data=xdata_numpy, columns=xdata_names)

    to_predict = {
        key: dict1[key][1] for key in xdata_names
    }  # takes the second value of all var in dict1, so function should return this value twice
    actual = collectValuesForLinRegr(
        df_xdata=df_xdata, to_predict=to_predict, threshold=0.1
    )
    # Verify
    assert actual == desired


def test_collectValuesForLinRegr_gets_input_values_above_threshold_so_should_return_two_different_values_for_var_above_threshold(
    dictForTest
):
    # Setup
    desired = {
        "glaz_area_distrib": [0.0, 0.0],
        "glazing_area": [0.0, 0.0],
        "height": [3.5, 7],  # diff values
        "roof_area": [147.0, 220.5],  # diff values
        "surf_area": [686, 686],
        "wall_area": [245.0, 245.0],
    }

    # Exercise
    dict_test = dictForTest
    to_predict = {
        "glaz_area_distrib": 0.0,
        "glazing_area": 0.0,
        "height": 4.5,
        "roof_area": 187.0,
        "surf_area": 686,
        "wall_area": 245.0,
    }
    xdata_names = dict_test["N1"]["xdata_names"]
    xdata_numpy = dict_test["N1"]["X"]["X_train"]
    df_xdata = pd.DataFrame(data=xdata_numpy, columns=xdata_names)

    actual = collectValuesForLinRegr(
        df_xdata=df_xdata, to_predict=to_predict, threshold=0.1
    )
    # Verify
    assert actual == desired


"""END Testing the function collectValuesForLinRegr"""


"""Testing the function makeDFforMultipleRegr"""
# Function that makes a dataframe for multiple interpolation of vars.
# INPUT: to_predict: dict with the values to predict Y, df_xdata: a dataframe with the train xdata, threshold: threshold to pass to the function collectValuesForLinRegr
def test_makeDFforMultipleRegr_returns_df_with_4_rows_when_2_vars_are_above_threshold(
    dictForTest
):
    # Setup
    desired = pd.DataFrame(
        [
            [686, 245, 147, 3.5, 0, 0],
            [686, 245, 147, 7, 0, 0],
            [686, 245, 220.5, 3.5, 0, 0],
            [686, 245, 220.5, 7, 0, 0],
        ],
        columns=[
            "surf_area",
            "wall_area",
            "roof_area",
            "height",
            "glazing_area",
            "glaz_area_distrib",
        ],
    )
    # Exercise
    dict_test = dictForTest
    to_predict = {
        "glaz_area_distrib": 0,
        "glazing_area": 0,
        "height": 4.5,
        "roof_area": 187,
        "surf_area": 686,
        "wall_area": 245,
    }
    xdata_names = dict_test["N1"]["xdata_names"]
    xdata_numpy = dict_test["N1"]["X"]["X_train"]
    df_xdata = pd.DataFrame(data=xdata_numpy, columns=xdata_names)

    actual = makeDFforMultipleRegr(
        to_predict=to_predict, df_xdata=df_xdata, threshold=0.1
    )[0]
    # Verify
    assert_frame_equal(actual, desired, check_dtype=False)


def test_makeDFforMultipleRegr_returns_string_that_says__All_var_are_below_the_threshold_i_e_no_interpolation__when_all_vars_are_below_threshold(
    dictForTest
):
    # Setup
    desired = "All vars are below the threshold - i.e. no interpolation"

    # Exercise
    dict_test = dictForTest
    to_predict = {
        "glaz_area_distrib": 0,
        "glazing_area": 0,
        "height": 3.5,
        "roof_area": 220,
        "surf_area": 686,
        "wall_area": 245,
    }
    xdata_names = dict_test["N1"]["xdata_names"]
    xdata_numpy = dict_test["N1"]["X"]["X_train"]
    df_xdata = pd.DataFrame(data=xdata_numpy, columns=xdata_names)

    actual = makeDFforMultipleRegr(
        to_predict=to_predict, df_xdata=df_xdata, threshold=0.1
    )
    # Verify
    assert actual == desired


"""END Testing the function makeDFforMultipleRegr"""

"""Testing the function findSlopes"""


def test_findSlopes_returns_correct_slopes(dictForTest):
    # Setup
    desired = [0, 0, -0.0755, 1.647, 0, 0]  # model.coef_

    # Exercise
    df_regr = pd.DataFrame(
        [
            [686, 245, 147, 3.5, 0, 0],
            [686, 245, 147, 7, 0, 0],
            [686, 245, 220.5, 3.5, 0, 0],
            [686, 245, 220.5, 7, 0, 0],
        ],
        columns=[
            "surf_area",
            "wall_area",
            "roof_area",
            "height",
            "glazing_area",
            "glaz_area_distrib",
        ],
    )
    dict_test = dictForTest
    predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]

    actual = findSlopes(df_xdata_regr=df_regr, predictor=predictor)[0][0]
    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)  #


"""END Testing the function findSlopes"""


"""Testing the function interpolate"""


def test_intepolate_returns_correct_values():
    # Y values generated by this function: f(x) = 2*x[0] - 3*x[1] + .5*x[2]

    # Setup
    desired = 1.25

    # Exercise
    df_regr = pd.DataFrame(
        data=[
            [1, 1, 1, -0.5],
            [2, 1, 1, 1.5],
            [1, 2, 2, -3],
            [2, 2, 2, -1],
            [1, 2, 1, -3.5],
            [2, 1, 2, 2],
            [1, 1, 2, 0],
            [2, 2, 1, -1.5],
        ],
        columns=["x1", "x2", "x3", "Y"],
    )

    slopes = [2, -3, 0.5]
    to_predict = {"x1": 1.8, "x2": 1.1, "x3": 1.9}
    to_predict_for_regr = {"x1": [1, 2], "x2": [1, 2], "x3": [1, 2]}

    actual = intepolate(
        df_regr=df_regr,
        slope=slopes,
        to_predict=to_predict,
        to_predict_for_regr=to_predict_for_regr,
    )

    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)  #


"""END Testing the function interpolate"""


"""Testing the function refineRF_PredictionByInterpolation"""


def test_refineRF_PredictionByInterpolation_returns_string_saying__All_vars_are_below_the_threshold_ie_no_interpolation__(
    dictForTest
):

    # Setup
    desired = "All vars are below the threshold - i.e. no interpolation"

    # Exercise
    dict_test = dictForTest
    xdata_names = dict_test["N1"]["xdata_names"]
    xdata_numpy = dict_test["N1"]["X"]["X_train"]
    to_predict = {
        "glaz_area_distrib": 0.0,
        "glazing_area": 0.0,
        "height": 3.5,
        "roof_area": 220.5,
        "surf_area": 686.0,
        "wall_area": 245.0,
    }
    predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]

    actual = refineRF_PredictionByInterpolation(
        xdata_names=xdata_names,
        xdata_numpy=xdata_numpy,
        to_predict_dict=to_predict,
        threshold=0.1,
        predictor=predictor,
    )

    # Verify
    assert desired == actual


def test_refineRF_PredictionByInterpolation_returns_correct_values(dictForTest):

    # Setup
    desired = 7.695

    # Exercise
    dict_test = dictForTest
    xdata_names = dict_test["N1"]["xdata_names"]
    xdata_numpy = dict_test["N1"]["X"]["X_train"]
    to_predict = {
        "glaz_area_distrib": 0.0,
        "glazing_area": 0.0,
        "height": 4.5,
        "roof_area": 220.5,
        "surf_area": 686.0,
        "wall_area": 245.0,
    }
    predictor = dict_test["N1"]["Y"]["heat_load"]["pred"]["forest"]["predictor"]

    actual = refineRF_PredictionByInterpolation(
        xdata_names=xdata_names,
        xdata_numpy=xdata_numpy,
        to_predict_dict=to_predict,
        threshold=0.1,
        predictor=predictor,
    )

    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)


"""END Testing the function refineRF_PredictionByInterpolation"""

import numpy as np
import pandas as pd
import unittest

import pytest
import numpy.testing as npt

from MLDA.corr_stats import stat_helperfunc as shf

from MLDA.corr_stats.corr_heatmap import (
    zero_replace,
    removeNan,
    corrType,
    corrMethodsExecute,
    checkdtypeOfColumns,
    correlation,
    findCorrPvalBasedAndNot,
    findCorrSelectMethod,
    screenCorrValuesBasedOnPvalue,
)


class TestFunctions(unittest.TestCase):
    def test0_corrType_out_is_string(self):
        result = corrType("cat", "cat")
        self.assertEqual(type("catcat"), type(result))

    def test1_corrType(self):
        result = corrType("cat", "cat")
        self.assertEqual("catcat", result)

    def test2_corrType(self):
        result = corrType("num", "cat")
        self.assertEqual("catnum", result)

    def test3_corrType(self):
        result = corrType("num", "num")
        self.assertEqual("numnum", result)

    def test1_corrMethodsExecute(self):
        result = corrMethodsExecute(
            "catcat", method_cc="Asym", method_cn="Omega_ols", method_nn="Spear"
        )
        self.assertEqual("Asym", result)

    def test2_corrMethodsExecute(self):
        result = corrMethodsExecute(
            "catnum",
            method_cc=["Asym"],
            method_cn=["Omega_ols"],
            method_nn=["Spear", "Pear", "MI_num"],
        )
        self.assertEqual(["Omega_ols"], result)

    def test3_corrMethodsExecute(self):
        result = corrMethodsExecute(
            "numnum",
            method_cc=["Asym"],
            method_cn=["Omega_ols"],
            method_nn=["Spear", "Pear", "MI_num"],
        )
        self.assertEqual(["Spear", "Pear", "MI_num"], result)

    def test0_corrValue_output_is_tuple_with_double_tuple(self):
        # testing value 1 of a list with 2 values
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        serie1, serie2 = tab1_input["num_police"], tab1_input["intimacy"]
        result = findCorrPvalBasedAndNot(serie1, serie2, "Asym")
        self.assertEqual("tuple", type(result).__name__)
        self.assertEqual("tuple", type(result[0]).__name__)
        self.assertEqual("tuple", type(result[1]).__name__)

    def test1_corrValue(self):
        # testing value 1 of a list with 2 values
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        serie1, serie2 = tab1_input["num_police"], tab1_input["intimacy"]
        result = findCorrPvalBasedAndNot(serie1, serie2, "Asym")
        self.assertAlmostEqual(0.019759842721107736, result[0][0], places=4)

    def test2_corrValue(self):
        # testing value 2 of a list with 2 values
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        serie1, serie2 = tab1_input["num_police"], tab1_input["intimacy"]
        result = findCorrPvalBasedAndNot(serie1, serie2, "Asym")
        self.assertAlmostEqual(0.006552725157981362, result[1][0], places=4)


class TestfindCorrSelectMethodFunction(unittest.TestCase):
    def test0_findCorrSelectMethod_output_is_numeric(self):
        # testing 2 cat series
        fifa = shf.spacesToUnderscore(
            pd.read_csv("/home/jesper/Work/macledan/input_files/data.csv")
        )
        catcol = [
            "Name",
            "Club",
            "Nationality",
            "Preferred_Foot",
            "Position",
            "Body_Type",
        ]
        numcol = []
        serie1, serie2 = fifa["Preferred_Foot"], fifa["Position"]
        result = findCorrSelectMethod(
            serie1,
            serie2,
            catcol,
            method_cc="Asym",
            method_cn="Omega_ols",
            method_nn="Spear",
        )
        self.assertEqual(
            True, isinstance(result[1][0], float) or isinstance(result[1][0], int)
        )

    def test1_findCorrSelectMethod(self):
        # testing 2 cat series
        fifa = shf.spacesToUnderscore(
            pd.read_csv("/home/jesper/Work/macledan/input_files/data.csv")
        )
        catcol = [
            "Name",
            "Club",
            "Nationality",
            "Preferred_Foot",
            "Position",
            "Body_Type",
        ]
        numcol = [
            "Age",
            "Overall",
            "Potential",
            "Crossing",
            "Finishing",
            "ShortPassing",
            "Dribbling",
            "LongPassing",
            "BallControl",
            "Acceleration",
            "SprintSpeed",
            "Agility",
            "Stamina",
        ]
        serie1, serie2 = fifa["Preferred_Foot"], fifa["Position"]
        result = findCorrSelectMethod(
            serie1,
            serie2,
            catcol,
            method_cc="Asym",
            method_cn="Omega_ols",
            method_nn="Spear",
        )
        self.assertAlmostEqual(0.212351469, result[1][0], places=4)

    def test2_findCorrSelectMethod(self):
        # testing 2 num series
        fifa = shf.spacesToUnderscore(
            pd.read_csv("/home/jesper/Work/macledan/input_files/data.csv")
        )
        catcol = ["Preferred_Foot", "Position", "Body_Type"]
        numcol = [
            "Dribbling",
            "LongPassing",
            "BallControl",
            "Acceleration",
            "SprintSpeed",
            "Agility",
            "Stamina",
        ]
        serie1, serie2 = fifa["BallControl"], fifa["Acceleration"]
        serie1, serie2 = removeNan(serie1, serie2)
        result = findCorrSelectMethod(
            serie1,
            serie2,
            catcol,
            method_cc="Asym",
            method_cn="Omega",
            method_nn="Spear",
        )
        self.assertAlmostEqual(0.54516, result[1][0], places=4)


"""Dataset used for test"""


@pytest.fixture()
def dfForTest():
    df_test = pd.read_excel("/home/jesper/Work/MLDA_app/MLDA/input_data/fortest_DF_shuf.xlsx")
    return df_test


"""END Dataset used for test"""


"""Testing checkdtypeOfColumns"""

# Test1
def test_checkdtypeOfColumns_returns_error_when_categorical_columns_not_category_choose_by_name(
    dfForTest
):
    # Setup

    desired = "The column 'global_warm_risk' is NOT category"

    # Exercise
    catcol = ["global_warm_risk", "intimacy"]
    numcol = []
    actual = checkdtypeOfColumns(catcol, numcol, dfForTest)

    # Verify
    assert actual == desired


# Test2 - same as test1 but with columns are choosed by integers
def test_checkdtypeOfColumns_returns_error_when_categorical_columns_not_category_choose_by_number(
    dfForTest
):
    # Setup

    desired = "The column '0' is NOT category"

    # Exercise
    catcol = [0, 1]
    numcol = []
    actual = checkdtypeOfColumns(catcol, numcol, dfForTest)

    # Verify
    assert actual == desired


# Test3 - Now the category should be OK
def test_checkdtypeOfColumns_returns_NOerror_when_categorical_columns_are_category_choose_by_name(
    dfForTest
):
    # Setup

    desired = "dtype of categorical columns: ['global_warm_risk', 'intimacy'] and numerical columns ['X1'] are ok"

    # Exercise
    catcol = ["global_warm_risk", "intimacy"]
    numcol = ["X1"]
    for entry in catcol:
        dfForTest[entry] = dfForTest[entry].astype("category")

    actual = checkdtypeOfColumns(catcol, numcol, dfForTest)

    # Verify
    assert actual == desired


# Test4
def test_checkdtypeOfColumns_returns_error_when_numeric_columns_not_float64_or_int64(
    dfForTest
):
    # Setup

    desired = "The column 'X1' is NOT float64/int64"

    # Exercise
    catcol = ["X1"]
    numcol = ["X1"]
    for entry in catcol:
        dfForTest[entry] = dfForTest[entry].astype("category")

    actual = checkdtypeOfColumns(catcol, numcol, dfForTest)

    # Verify
    assert actual == desired


# Test5 - Now the cat/num columns should be OK
def test_checkdtypeOfColumns_returns_NOerror_when_numerical_columns_are_float64ORINT64_choose_by_number(
    dfForTest
):
    # Setup

    desired = "dtype of categorical columns: [0, 1] and numerical columns [7, 8] are ok"

    # Exercise
    catcol = [0, 1]
    numcol = [7, 8]
    for entry in catcol:
        dfForTest.iloc[:, entry] = dfForTest.iloc[:, entry].astype("category")

    actual = checkdtypeOfColumns(catcol, numcol, dfForTest)

    # Verify
    print("test5")
    assert actual == desired


"""END Testing checkdtypeOfColumns"""


"""Testing findCorrPvalBasedAndNot"""


def test_findCorrPvalBasedAndNot_returns_correct_dtype_and_correct_result_catnum(
    dfForTest
):
    # Setup
    desired1 = 2
    desired2 = ((0.451751, 0.01591), (0.451751, 0.01591))
    # Exercise
    serie1, serie2 = dfForTest["group"], dfForTest["weight"]
    serie1, serie2 = removeNan(serie1, serie2)
    actual = findCorrPvalBasedAndNot(serie1, serie2, "Omega")

    # Verify
    assert len(actual) == desired1  # length of outer array
    assert len(actual[0]) == desired1  # length of inner array
    npt.assert_array_almost_equal(actual, desired2)


def test_findCorrPvalBasedAndNot_returns_correct_dtype_and_correct_result_catcat_sym(
    dfForTest
):
    # Setup
    desired1 = 2
    desired2 = ((0.145864, 0.893955), (0.145864, 0.893955))
    # Exercise
    serie1, serie2 = dfForTest["intimacy"], dfForTest["num_police"]
    serie1, serie2 = removeNan(serie1, serie2)
    actual = findCorrPvalBasedAndNot(serie1, serie2, "Cramer_V")

    # Verify
    assert len(actual) == desired1  # length of outer array
    assert len(actual[0]) == desired1  # length of inner array
    npt.assert_array_almost_equal(actual, desired2)


def test_findCorrPvalBasedAndNot_returns_correct_dtype_and_correct_result_catcat_asym(
    dfForTest
):
    # Setup
    desired1 = 2
    desired2 = (
        (0.006285953589698483, 0.124547),
        (0.002752174685457338, 0.091512),
    )
    # Exercise
    serie1, serie2 = dfForTest["global_warm_risk"], dfForTest["gender"]
    serie1, serie2 = removeNan(serie1, serie2)
    actual = findCorrPvalBasedAndNot(serie1, serie2, "Asym")

    # Verify
    assert len(actual) == desired1  # length of outer array
    assert len(actual[0]) == desired1  # length of inner array
    npt.assert_array_almost_equal(actual, desired2)


"""END Testing findCorrPvalBasedAndNot"""

"""Testing findCorrSelectMethod"""


def test_findCorrSelectMethod_returns_correct_dtype_and_correct_result_catcat_asym(
    dfForTest
):
    # Setup
    desired1 = 2
    desired2 = (
        (0.006285953589698483, 0.124547),
        (0.002752174685457338, 0.091512),
    )
    # Exercise
    serie1, serie2 = dfForTest["global_warm_risk"], dfForTest["gender"]
    serie1, serie2 = removeNan(serie1, serie2)
    catcol = ["heyhey", "global_warm_risk", "gender", "heysan"]
    actual = findCorrSelectMethod(serie1, serie2, catcol, CI=1, method_cc="Asym")

    # Verify
    assert len(actual) == desired1  # length of outer array
    assert len(actual[0]) == desired1  # length of inner array
    npt.assert_array_almost_equal(actual, desired2)


"""END Testing findCorrPvalBasedAndNot"""

"""Testing screenCorrValuesBasedOnPvalue"""


def test_screenCorrValuesBasedOnPvalue_returns_correct_dtype_and_result_based_on_CI(
    dfForTest
):
    # Setup
    desired1 = 2
    desired2 = (("p > CI", 0.2), (0.4, 0.09))

    # Exercise
    actual = screenCorrValuesBasedOnPvalue(((0.2, 0.2), (0.4, 0.09)), CI=0.1)

    # Verify
    assert len(actual) == desired1  # length of outer array
    assert len(actual[0]) == desired1  # length of inner array
    assert actual == desired2


"""END Testing screenCorrValuesBasedOnPvalue"""


""" Testing correlation"""


def test_correlation_returns_correct_corr_values_in_matrix_corr_dfault_methods(dfForTest):
    # Setup
    desired1 = 0.0926
    desired2 = 0.1459
    desired3 = 0.45175
    desired4 = -0.2558
    desired5 = 0.622

    # Exercise
    catcols = ["global_warm_risk", "intimacy", "gender", "group", "num_police"]
    numcols = ["X1", "weight", "X3", "Y1"]
    actual = correlation(
        dfForTest,
        catcols,
        numcols,
        CI=1,
        method_cc="Cramer_V",
        method_cn="Omega",
        method_nn="Spearmann",
    )[0]

    # Verify
    npt.assert_almost_equal(actual.iloc[0][2], desired1, decimal=4)
    npt.assert_almost_equal(actual.iloc[1][4], desired2, decimal=4)
    npt.assert_almost_equal(actual.iloc[3][6], desired3, decimal=4)
    npt.assert_almost_equal(actual.iloc[5][7], desired4, decimal=4)
    npt.assert_almost_equal(actual.iloc[5][8], desired5, decimal=4)


def test_correlation_returns_correct_p_values_in_matrix_corr_dfault_methods(dfForTest):
    # Setup
    desired1 = 0.0002269
    desired2 = 0.8940
    desired4 = 0.01591
    desired5 = .001
    desired6 = 0.001

    # Exercise
    catcols = ["global_warm_risk", "intimacy", "President", "gender", "group", "Color", "num_police"]
    numcols = ["X1", "weight", "X3", "Y1"]
    actual = correlation(
        dfForTest,
        catcols,
        numcols,
        CI=1,
        method_cc="Cramer_V",
        method_cn="Omega",
        method_nn="Spearmann",
    )[1]

    # Verify
    npt.assert_almost_equal(actual.iloc[0][3], desired1, decimal=6) # global_warm_risk vs gender
    npt.assert_almost_equal(actual.iloc[1][6], desired2, decimal=4) # intimacy vs num_police
    assert actual.iloc[2][5] < desired5 # President vs Color
    npt.assert_almost_equal(actual.iloc[4][8], desired4, decimal=4) # group vs weight
    assert actual.iloc[7][9] < desired5 # X1 vs X3
    assert actual.iloc[7][10] < desired6 # X1 vs Y1


def test_correlation_returns_correct_corr_values_in_matrix_corr_asym_method(dfForTest):
    # Setup
    desired1 = 0.0062859
    desired2 = 0.00275217
    desired3 = 0.808693
    desired4 = 0.323283


    # Exercise
    catcols = ["global_warm_risk", "intimacy", "President", "gender", "group", "Color", "num_police"]
    numcols = ["X1", "weight", "X3", "Y1"]
    actual = correlation(
        dfForTest,
        catcols,
        numcols,
        CI=1,
        method_cc="Asym",
        method_cn="Omega",
        method_nn="Spearmann",
    )[0]

    # Verify
    npt.assert_almost_equal(actual.iloc[0][3], desired1, decimal=4) # global_warm_risk vs gender
    npt.assert_almost_equal(actual.iloc[3][0], desired2, decimal=4) # gender vs global_warm_risk
    npt.assert_almost_equal(actual.iloc[2][5], desired3, decimal=4) # President vs Color. If we know president we know the color
    npt.assert_almost_equal(actual.iloc[5][2], desired4, decimal=4) # Color vs  President. If the know the color we are not sure about the President


"""END Testing correlation"""


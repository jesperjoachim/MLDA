import unittest

import pytest
import numpy.testing as npt

from MLDA.corr_stats.stat_functions import *


class TestStatFunctions(unittest.TestCase):
    def test_cramers_v_correct_cramers_v_value(self):
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        result = cramers_v(tab1_input["intimacy"], tab1_input["num_police"])[0]
        self.assertAlmostEqual(0.1458643, result, places=4)

    def test_cramers_v_correct_pValue(self):
        """p-value (0.0002269) is taken from https://bookdown.org/ripberjt/qrmbook/association-of-variables.html - 6 Association of Variables"""
        input = pd.read_excel(
            "/home/jesper/Work/MLDA_app/MLDA/input_data/globalwarm_risks1to5_women_men.xlsx"
        )
        result = cramers_v(input["global_warm_risk"], input["gender"])[1]
        self.assertAlmostEqual(0.000226946932, result, places=4)

    def test1_u_y(self):
        """corr-value (0.01975984) is taken from http://www.statystyka.c0.pl/blog/association-measures-for-categorical-data.html - Association measures for categorical data"""
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        result = u_y(tab1_input["num_police"], tab1_input["intimacy"])
        self.assertAlmostEqual(0.01975984, result, places=4)

    def test2_u_y(self):
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        result = u_y(tab1_input["intimacy"], tab1_input["num_police"])
        self.assertAlmostEqual(0.00655273, result, places=4)

    def test_MI_cat(self):
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        result = MI_cat(tab1_input["num_police"], tab1_input["intimacy"])
        self.assertAlmostEqual(0.00499870999792, result, places=4)

    def test_omega_ols(self):
        """corr-value (0.45175) is taken from https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/ - Four Ways to Conduct One-Way ANOVA with Python."""
        plant_data = pd.read_csv(
            "/home/jesper/Work/macledan/input_files/PlantGrowth.csv"
        )
        result = omega_ols(plant_data["group"], plant_data["weight"])[0]
        self.assertAlmostEqual(0.45175086717124396, result, places=6)

    def test_MI_num(self):
        np.random.seed(0)
        X = np.random.rand(1000, 3)
        x2 = X[:, 1]
        y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)
        serie1, serie2 = pd.Series(x2), pd.Series(y)
        result = MI_num(serie1, serie2)
        self.assertAlmostEqual(0.86235025520, result, places=5)

    def test_spearmann1(self):
        # Test with p-value < 0.1
        user_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "y1", "y2"]
        # we name the columns/header as defined in user_cols and we skip the first row
        energy = pd.read_excel(
            "/home/jesper/Work/macledan/input_files/ENB2012_data.xlsx",
            header=None,
            names=user_cols,
            skiprows=1,
        )
        result = spearmann(energy["x1"], energy["x5"])[0]
        self.assertAlmostEqual(0.869048, result, places=4)

    def test_pearson1(self):
        # Test with p-value < 0.1
        user_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "y1", "y2"]
        # we name the columns/header as defined in user_cols and we skip the first row
        energy = pd.read_excel(
            "/home/jesper/Work/macledan/input_files/ENB2012_data.xlsx",
            header=None,
            names=user_cols,
            skiprows=1,
        )
        result = pearson(energy["x1"], energy["x5"])[0]
        self.assertAlmostEqual(0.827747316838428, result, places=4)


# ###----------------------------------------------------------------###

# Test dataframes
df_test = pd.read_excel(
    "/home/jesper/Work/MLDA_app/MLDA/input_data/fortest_DF_shuf.xlsx"
)
input = pd.read_excel(
    "/home/jesper/Work/MLDA_app/MLDA/input_data/globalwarm_risks1to5_women_men.xlsx"
)


# def test_cramersV_out_is_zero_when_crosstab_dim_is_below_2x2_for_testing_purpose_only():
#     # Setup
#     desired = (np.nan, np.nan)

#     # Exercise
#     serie1, serie2 = df_test["global_warm_risk"], df_test["intimacy"]
#     serie1, serie2 = removeNan(serie1, serie2)
#     actual = cramers_v(serie1, serie2)
#     # Verify
#     npt.assert_almost_equal(actual, desired, decimal=4)


# # Testing calcMeanAndStdDev(method, serie1, serie2)
# def test_calcMeanAndStdDev_with_Asym_returns_MeancorrVal_and_stdcorrVal_from_dftest_globalwarmrisk_gender():
#     # Setup
#     np.random.seed(0)
#     desired1, desired2 = 0.0008694019885874294, 0.00045475323268101103

#     # Exercise
#     serie1, serie2 = df_test["global_warm_risk"], df_test["gender"]
#     serie1, serie2 = removeNan(serie1, serie2)
#     actual = calcMeanAndStdDev(serie1, serie2, "Asym")

#     # Verify
#     npt.assert_almost_equal(actual[0], desired1, decimal=4)
#     npt.assert_almost_equal(actual[1], desired2, decimal=4)


# # Testing calcMeanAndStdDev(method, serie1, serie2)
# def test_calcMeanAndStdDev_with_MIcat_returns_MeancorrVal_and_stdcorrVal_from_dftest_globalwarmrisk_gender():
#     # Setup
#     np.random.seed(0)
#     desired1, desired2 = 0.00027228204089949635, 0.0002287275600370245

#     # Exercise
#     serie1, serie2 = df_test["global_warm_risk"], df_test["gender"]
#     serie1, serie2 = removeNan(serie1, serie2)
#     actual = calcMeanAndStdDev(serie1, serie2, "MI_cat")

#     # Verify
#     npt.assert_almost_equal(actual[0], desired1, decimal=4)
#     npt.assert_almost_equal(actual[1], desired2, decimal=4)


# Testing calcCorrNonP(serie1, serie2, method)
def test_calcCorrNonP_returns_two_value_tuple_when_method_is_asym():
    # Setup
    desired1 = "tuple"
    desired2 = 2
    desired3 = "float64"

    # Exercise
    actual = calcCorrNonP(df_test["global_warm_risk"], df_test["gender"], "Asym")

    # Verify
    assert type(actual).__name__ == desired1
    assert len(actual) == desired2
    assert type(actual[0]).__name__ == desired3


"""Testing mimicPvalueCalc(mean_and_stdev, corr_value)"""


def test_mimicPvalueCalc_returns_singlevalue_float64():
    # Setup
    desired1 = "float"
    desired2 = 0.1

    # Exercise
    actual = mimicPvalueCalc((0.2, 0.1), 1.2)

    # Verify
    assert type(actual).__name__ == desired1
    assert actual == desired2


def test_mimicPvalueCalc_returns_handles_division_by_zero():
    # Setup
    desired = "nan"

    # Exercise
    actual = mimicPvalueCalc((0.20000000000001, 0.1), 0.2)

    # Verify
    assert str(actual) == desired


"""END Testing mimicPvalueCalc"""


# Testing calcCorrAndMimicP(serie1, serie2, method)
def test_calcCorrAndMimicP_returns_double_two_different_value_tuple_from_asym():
    # Setup
    desired1 = "tuple"
    desired2 = ((0.006285953589698483, 0.1245), (0.002752174685457338, 0.0915))

    # Exercise
    serie1, serie2 = df_test["global_warm_risk"], df_test["gender"]
    actual = calcCorrAndMimicP(serie1, serie2, "Asym")

    # Verify
    assert type(actual).__name__ == desired1
    assert type(actual[0]).__name__ == desired1
    npt.assert_almost_equal(actual, desired2, decimal=4)


def test_calcCorrAndMimicP_returns_double_two_same_value_tuple_from_sym():
    # Setup
    desired1 = "tuple"
    desired2 = ((0.0018411497470424143, 0.1245), (0.0018411497470424143, 0.1245))

    # Exercise
    serie1, serie2 = df_test["global_warm_risk"], df_test["gender"]
    actual = calcCorrAndMimicP(serie1, serie2, "MI_cat")

    # Verify
    assert type(actual).__name__ == desired1
    assert type(actual[0]).__name__ == desired1
    npt.assert_almost_equal(actual, desired2, decimal=4)


"""Dataset used for test"""


@pytest.fixture()
def dfForTest():
    df_test = pd.read_excel(
        "/home/jesper/Work/MLDA_app/MLDA/input_data/fortest_DF_shuf.xlsx"
    )
    return df_test


"""END Dataset used for test"""


"""Testing calcCorrAndP"""


def test_calcCorrAndP_returns_correct_datatype_and_values_numnum(dfForTest):
    # Setup
    desired1 = 2  # length of array
    desired2 = (0.6221347, 1.771230e-83)
    # Exercise
    serie1, serie2 = dfForTest["X1"], dfForTest["Y1"]
    serie1, serie2 = removeNan(serie1, serie2)
    actual = calcCorrAndP(serie1, serie2, "Spear")

    # Verify
    assert len(actual) == desired1
    npt.assert_array_almost_equal(actual, desired2)


def test_calcCorrAndP_returns_correct_datatype_and_values_catcat(dfForTest):
    # Setup
    desired1 = 2  # length of array
    desired2 = (0.145864, 0.893955)
    # Exercise
    serie1, serie2 = dfForTest["intimacy"], dfForTest["num_police"]
    serie1, serie2 = removeNan(serie1, serie2)
    actual = calcCorrAndP(serie1, serie2, "Cramer_V")

    # Verify
    assert len(actual) == desired1
    npt.assert_array_almost_equal(actual, desired2)


def test_calcCorrAndP_returns_correct_datatype_and_values_catnum(dfForTest):
    # Setup
    desired1 = 2  # length of array
    desired2 = (0.451751, 0.01591)
    # Exercise
    serie1, serie2 = dfForTest["group"], dfForTest["weight"]
    serie1, serie2 = removeNan(serie1, serie2)
    actual = calcCorrAndP(serie1, serie2, "Omega")

    # Verify
    assert len(actual) == desired1
    npt.assert_array_almost_equal(actual, desired2)


"""END Testing calcCorrAndP"""

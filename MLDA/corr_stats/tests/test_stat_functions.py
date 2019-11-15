import unittest
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
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        result = u_y(tab1_input["num_police"], tab1_input["intimacy"])[0]
        self.assertAlmostEqual(0.01975984, result, places=4)

    def test2_u_y(self):
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        result = u_y(tab1_input["intimacy"], tab1_input["num_police"])[0]
        self.assertAlmostEqual(0.00655273, result, places=4)

    def test_MI_cat(self):
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        result = MI_cat(tab1_input["num_police"], tab1_input["intimacy"])[0]
        self.assertAlmostEqual(0.00499870999792, result, places=4)

    def test_omega_ols(self):
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
        result = MI_num(serie1, serie2)[0]
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


###----------------------------------------------------------------###

# Test dataframes
df_test = pd.read_excel("/home/jesper/Work/macledan/input_files/test_DF.xlsx")
input = pd.read_excel(
    "/home/jesper/Work/MLDA_app/MLDA/input_data/globalwarm_risks1to5_women_men.xlsx"
)


def test_cramersV_out_is_zero_when_crosstab_dim_is_below_2x2_for_testing_purpose_only():
    # Setup
    desired = (np.nan, np.nan)

    # Exercise
    serie1, serie2 = df_test["global_warm_risk"], df_test["intimacy"]
    serie1, serie2 = removeNan(serie1, serie2)
    actual = cramers_v(serie1, serie2)
    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)


# Testing calcMeanAndStdDev(method, serie1, serie2)
def test_calcMeanAndStdDev_with_Asym_returns_MeancorrVal_and_stdcorrVal_from_dftest_globalwarmrisk_gender():
    # Setup
    np.random.seed(0)
    desired1, desired2 = 0.0008694019885874294, 0.00045475323268101103

    # Exercise
    serie1, serie2 = df_test["global_warm_risk"], df_test["gender"]
    serie1, serie2 = removeNan(serie1, serie2)
    actual = calcMeanAndStdDev("Asym", serie1, serie2)

    # Verify
    npt.assert_almost_equal(actual[0], desired1, decimal=4)
    npt.assert_almost_equal(actual[1], desired2, decimal=4)


# Testing calcMeanAndStdDev(method, serie1, serie2)
def test_calcMeanAndStdDev_with_MIcat_returns_MeancorrVal_and_stdcorrVal_from_dftest_globalwarmrisk_gender():
    # Setup
    np.random.seed(0)
    desired1, desired2 = 0.00027228204089949635, 0.0002287275600370245

    # Exercise
    serie1, serie2 = df_test["global_warm_risk"], df_test["gender"]
    serie1, serie2 = removeNan(serie1, serie2)
    actual = calcMeanAndStdDev("MI_cat", serie1, serie2)

    # Verify
    npt.assert_almost_equal(actual[0], desired1, decimal=4)
    npt.assert_almost_equal(actual[1], desired2, decimal=4)


# Testing evalSignificance(method, serie1, serie2, CI=0.1, std_val=1.5)
def test_evalSignificance_with_Asym_returns_corrVal_from_dftest_globalwarmrisk_gender():
    # Setup
    np.random.seed(0)
    desired = 0.006285953589698483

    # Exercise
    serie1, serie2 = df_test["global_warm_risk"], df_test["gender"]
    actual = evalSignificance("Asym", serie1, serie2)

    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)


# Testing evalSignificance(method, serie1, serie2, CI=0.1, std_val=1.5)
def test_evalSignificance_with_MI_num_returns_corrVal_from_dftest_globalwarmrisk_gender():
    # Setup
    np.random.seed(0)
    desired = 0.0018411497470424143

    # Exercise
    serie1, serie2 = df_test["global_warm_risk"], df_test["gender"]
    actual = evalSignificance("MI_cat", serie1, serie2)

    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)


# Testing evalSignificance(method, serie1, serie2, CI=0.1, std_val=5)
def test_evalSignificance_with_Asym_returns_Corr_is_Insignificant_from_dftest_globalwarmrisk_gender():
    # Setup
    np.random.seed(0)
    desired = "Corr is Insignificant"

    # Exercise
    serie1, serie2 = df_test["global_warm_risk"], df_test["gender"]
    actual = evalSignificance("Asym", serie1, serie2, CI=0.01)

    # Verify
    assert actual == desired


# Testing evalSignificance(method, serie1, serie2, CI=0.1, std_val=5)
def test_evalSignificance_with_MIcat_returns_Corr_is_Insignificant_from_dftest_globalwarmrisk_gender():
    # Setup
    np.random.seed(0)
    desired = "Corr is Insignificant"

    # Exercise
    serie1, serie2 = df_test["global_warm_risk"], df_test["gender"]
    actual = evalSignificance("MI_cat", serie1, serie2, CI=0.01)

    # Verify
    assert actual == desired


# Testing evalSignificance(method, serie1, serie2, CI=0.1, std_val=1.5)
def test_evalSignificance_with_spearmann_returns_corrVal_from_dftest_X1_Y1():
    # Setup
    desired = 0.62227

    # Exercise
    serie1, serie2 = removeNan(df_test["X1"], df_test["Y1"])
    actual = evalSignificance("Spearmann", serie1, serie2)

    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)


def test_evalSignificance_with_spearmann_returns_NO_corrVal_from_dftest_X1_weight():
    # Setup
    desired = "p-value > CI"

    # Exercise
    actual = evalSignificance("Spearmann", df_test["X1"], df_test["weight"])

    # Verify
    assert actual == desired


def test_evalSignificance_with_CramersV_returns_corrVal_from_input_globwarmrisk():
    # Setup
    np.random.seed(0)
    desired = 0.092582202

    # Exercise
    actual = evalSignificance(
        "Cramer_V", df_test["global_warm_risk"], df_test["gender"]
    )

    # Verify
    npt.assert_almost_equal(actual, desired, decimal=4)


def test_evalSignificance_with_CramersV_NOT_returns_corrVal_from_input_globwarmrisk():
    # Setup
    np.random.seed(0)
    desired = "p-value > CI"

    # Exercise
    actual = evalSignificance(
        "Cramer_V", df_test["global_warm_risk"], df_test["gender"], CI=0.00001
    )

    # Verify
    assert actual == desired

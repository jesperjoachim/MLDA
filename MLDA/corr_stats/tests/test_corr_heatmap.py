import numpy as np
import pandas as pd
import unittest
import numpy.testing as npt

from MLDA.corr_stats import stat_helperfunc as shf

from MLDA.corr_stats.corr_heatmap import (
    zero_replace,
    removeNan,
    corrType,
    corrMethodsExecute,
    corrValue,
    findCorr,
    correlation,
)


class TestFunctions(unittest.TestCase):
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
            "catcat",
            method_cc=["Asym"],
            method_cn=["Omega_ols"],
            method_nn=["Spear", "Pear", "MI_num"],
        )
        self.assertEqual(["Asym"], result)

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

    def test1_corrValue(self):
        # testing value 1 of a list with 2 values
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        serie1, serie2 = tab1_input["num_police"], tab1_input["intimacy"]
        result = corrValue(serie1, serie2, ["Asym"])
        self.assertAlmostEqual(0.019759842721107736, result[0], places=4)

    def test2_corrValue(self):
        # testing value 2 of a list with 2 values
        tab1_input = pd.read_excel("/home/jesper/Work/macledan/input_files/tab1.xlsx")
        serie1, serie2 = tab1_input["num_police"], tab1_input["intimacy"]
        result = corrValue(serie1, serie2, ["Asym"])
        self.assertAlmostEqual(0.006552725157981362, result[1], places=4)


class TestFindCorrFunctions(unittest.TestCase):
    def test1_findCorr(self):
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
        result = findCorr(
            serie1,
            serie2,
            catcol,
            method_cc=["Asym"],
            method_cn=["Omega_ols"],
            method_nn=["Spear", "Pear", "MI_num"],
        )
        self.assertAlmostEqual(0.212351469, result[1], places=4)

    def test2_findCorr(self):
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
        result = findCorr(
            serie1,
            serie2,
            catcol,
            method_cc=["Asym"],
            method_cn=["Omega"],
            method_nn=["Spear", "Pear", "MI_num"],
        )
        self.assertAlmostEqual(0.6757374207, result[1], places=4)

    # def test_correlation_test_df_with_3_different_datasets(self):
    #     # Setup
    #     df_test = pd.read_excel("/home/jesper/Work/macledan/input_files/test_DF.xlsx")

    #     # Exercise
    #     actual = correlation(
    #         df_test,
    #         ["intimacy", "group", "num_police"],
    #         ["X1", "weight", "X3", "Y1"],
    #         method_nn=["Spear"],
    #     )

    #     # Verify
    #     desired = np.array(
    #         [
    #             [
    #                 1,
    #                 0.3028956049403181,
    #                 0.006552725157981362,
    #                 0.4403605599541179,
    #                 0.2942711008313611,
    #                 0.0,
    #                 0.07091174645457968,
    #             ],
    #             [
    #                 0.5166591625747339,
    #                 1,
    #                 0.14830624374966814,
    #                 0.9044838623462509,
    #                 0.45175086717124396,
    #                 0.0,
    #                 0.4284339218917304,
    #             ],
    #             [
    #                 0.019759842721107736,
    #                 0.2393220043271394,
    #                 1,
    #                 0.4192617528903058,
    #                 0.0,
    #                 0.20210066874237836,
    #                 0.6027887351700807,
    #             ],
    #             [
    #                 0.4403605599541179,
    #                 0.9044838623462509,
    #                 0.4192617528903058,
    #                 1,
    #                 -0.3157755946258407,
    #                 -0.25580533357509605,
    #                 0.6221346925113646,
    #             ],
    #             [
    #                 0.2942711008313611,
    #                 0.45175086717124396,
    #                 0.0,
    #                 -0.3157755946258407,
    #                 1,
    #                 0,
    #                 0,
    #             ],
    #             [
    #                 0.0,
    #                 0.0,
    #                 0.20210066874237836,
    #                 -0.25580533357509605,
    #                 0,
    #                 1,
    #                 0.47145764701389403,
    #             ],
    #             [
    #                 0.07091174645457968,
    #                 0.4284339218917304,
    #                 0.6027887351700807,
    #                 0.6221346925113646,
    #                 0,
    #                 0.47145764701389403,
    #                 1,
    #             ],
    #         ]
    #     )
    #     npt.assert_almost_equal(actual, desired, decimal=4)

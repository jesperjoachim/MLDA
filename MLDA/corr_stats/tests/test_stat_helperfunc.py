import unittest
import numpy.testing as npt
import pandas as pd

from corr_stats.stat_helperfunc import *

#### NOT FINISHED
# class TestStatFunctions(unittest.TestCase):
#     def test_stripChar(self):
#         data = pd.read_csv("/home/jesper/Work/macledan/input_files/data.csv")
#         result_series = data["Value"].apply(stripChar, args=["M"])
#         npt.assert_almost_equal(result_series.head(5), 0, decimal=10)

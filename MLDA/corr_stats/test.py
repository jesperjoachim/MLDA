# from MLDA.corr_stats import corr_heatmap
def test(x, y):
    x = x + 4
    y = y * y
    return x, None


print(type(test(2, 3)), test(2, 3)[0])
# import pandas as pd
# import stat_helperfunc as shf

# data = pd.read_excel("/home/jesper/Work/macledan/input_files/test_DF.xlsx")

# print(data)

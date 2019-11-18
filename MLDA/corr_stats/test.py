# from MLDA.corr_stats import corr_heatmap
import pandas as pd


def test(x, y):
    x = x + 4
    y = y * y
    return x, None


# print(type(test(2, 3)), test(2, 3)[0])
# import pandas as pd
# import stat_helperfunc as shf

data = pd.read_excel("/home/jesper/Work/macledan/input_files/test_DF.xlsx")

# print(dir(data["gender"]))
# print(dir(data.iloc[:, [1]]))

# print(data["gender"].dtype.name)
# print()
# print(data.iloc[:, [1]].dtypes.dtype.name)

# print(data)
# cat = []
# num = ["hey"]
# print(str((cat + num)[0]).isnumeric())

# catcol = [0, 1]
# numcol = [5, 6]

# for entry in catcol:
#     # print(data.iloc[:, entry])
#     print(data.iloc[:, entry].dtype.name)
#     data.iloc[:, entry] = data.iloc[:, entry].astype("category")
#     print(data.iloc[:, entry].dtype.name)

# x = 13

# print(isinstance(x, float) or isinstance(x, int))


def multiply(x, y):
    return x * y


values = multiply(3, 4), multiply(4, 5)
values = values, values
print(values)

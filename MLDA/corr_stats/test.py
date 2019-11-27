# from MLDA.corr_stats import corr_heatmap
import pandas as pd
import numpy as np


def test(x, y):
    x = x + 4
    y = y * y
    return x, None


# print(type(test(2, 3)), test(2, 3)[0])
# import pandas as pd
# import stat_helperfunc as shf


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


# def multiply(x, y):
#     return x * y


# values = multiply(3, 4), multiply(4, 5)
# values = values, values
# print(values)

# actual = np.NaN
# print(str(actual))
# print(type(actual).__name__)

# Construct a dataframe

# data_orig = [[416, 121], [335, 2], [112, 1]]
# df_orig = pd.DataFrame(
#     data=data_orig, index=["Clinton", "Dole", "Perot"], columns=["white", "black"]
# )
# print(df_orig)

data_ordered = [
    ["male", 25, "high"],
    ["female", 15, "medium"],
    ["male", 35, "low"],
    ["female", 25, "high"],
    ["fmale", 21, "medium"],
]

df_ordered = pd.DataFrame(data=data_ordered, columns=["gender", "age", "height"])

# print(df_ordered)

data_right = [[2, "no"], [1, "yes"], [0, "no"]]

df_right = pd.DataFrame(data=data_right, columns=["children", "married"])
# print(df_right)

result = pd.concat([df_ordered, df_right], axis=1, ignore_index=True)
# print(result)
# print()
# df_shuffle = df_ordered.sample(frac=1)
# print(df_shuffle)

df_ordered = pd.read_excel("/home/jesper/Work/MLDA_app/MLDA/input_data/test_DF.xlsx")

# Index(['global_warm_risk', 'intimacy', 'President', 'gender', 'group', 'Color', 'num_police', 'X1', 'weight', 'X3', 'Y1']

df_inti_numpol = (
    df_ordered[["intimacy", "num_police"]].dropna().sample(frac=1, replace=True)
)
df_glo_gen = (
    df_ordered[["global_warm_risk", "gender"]].dropna().sample(frac=1, replace=True)
)
df_pres_col = df_ordered[["President", "Color"]].dropna().sample(frac=1)
df_gro_weig = df_ordered[["group", "weight"]].dropna().sample(frac=1)
df_x1x3y1 = df_ordered[["X1", "X3", "Y1"]].dropna().sample(frac=1)

df_test_shuffled = pd.concat([df_inti_numpol, df_glo_gen], axis=1, ignore_index=True)

# print(df_test_shuffled)

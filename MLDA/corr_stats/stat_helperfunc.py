import numpy as np
import pandas as pd

"""Functions for data analysis"""


def spacesToUnderscore(dataframe):
    """Change the spaces in column names to underscores"""
    new_columns = [string.replace(" ", "_") for string in dataframe.columns]
    dataframe.columns = new_columns
    return dataframe


def stripChar(string, char_to_remove=[]):
    # The syntax when passing in a df or series is:
    # data["Value"] = data["Value"].apply(stripChar, args=[["M", "€", "K"]])
    # Note when using apply the first arg for the chosen func is the object in front of .apply
    try:
        for char in char_to_remove:
            if char in string:
                string = string.replace(char, "")
        return string
    except:
        return string


def toFloatAndMultiply(digit_str, multiply=1):
    try:
        num = float(digit_str)
        num *= multiply
        return num
    except:
        return digit_str


def serieStripChar(serie, char_to_remove=[]):
    serie = serie.apply(lambda x: stripChar(x))


def wage_split(x):
    try:
        if str(x).isdigit():
            return x
        else:
            return int(x.split("K")[0][1:])
    except:
        return x


def value_split(x):
    try:
        if str(x).isdigit():
            return x
        elif "K" in x:
            return float(x.split("K")[0][1:]) / 1000
        elif "M" in x:
            return float(x.split("M")[0][1:])
    except:
        return x


# data["Wage"] = data["Wage"].apply(lambda x: wage_split(x))
# data["Value"] = data["Value"].apply(lambda x: value_split(x))


def convert(data, to):
    converted = None
    if to == "array":
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == "list":
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == "dataframe":
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError(
            "cannot handle data conversion of type: {} to {}".format(type(data), to)
        )
    else:
        return converted


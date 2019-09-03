"""
Utils for interoperability with Java/JVM.
This module contains things that can be reloaded.
"""

from jpype import JArray, JInt


def df_as_java_array(df, col):
    series = df[col]
    if not isinstance(col, str):  # if list of columns
        pass#series = series.T
    arr = series.to_numpy()
    num_dims = len(arr.shape)
    return JArray(JInt, num_dims)(arr)

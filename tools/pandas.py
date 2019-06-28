def binarize(df, cols=None, thres=None, inplace=True):
    if not inplace:
        raise NotImplementedError()
    cols = cols or df.columns
    for col in df:
        df[col] = (df[col] > thres) if thres else (df[col] > df[col].mean())
    return df
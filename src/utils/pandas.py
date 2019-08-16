
def binarize(df, cols=None, thres=None, inplace=True):
    if not inplace:
        raise NotImplementedError()
    cols = cols or df.columns
    for col in df:
        df[col] = (df[col] > thres) if thres else (df[col] > df[col].mean())
    return df


def df_map_columns(data, cols, lookup, col_names=None, replace_cols=True):
    merged = data.copy()
    col_names = col_names or cols
    for col, col_new_name in zip(cols, col_names):
        merged_col = df_map_column(merged, col, lookup)

        if replace_cols:
            merged.pop(col)
            
        merged[col_new_name] = merged_col
    
    return merged


def df_map_column(data, col, lookup):
    merged = data[[col]].merge(lookup, how='left', left_on=col, right_index=True)#, suffixes=suffixes)
    return merged[lookup.name]


def series_append_new_rows(series, other):
    idx_diff = other.index.difference(series)
    return series.append(other[idx_diff])


def series_freq(series):
    return series.value_counts(dropna=False).sort_values(ascending=False)

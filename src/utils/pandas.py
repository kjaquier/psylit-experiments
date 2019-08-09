
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
        merged_col = merged[[col]].merge(lookup, how='left', left_on=col, right_index=True)#, suffixes=suffixes)

        if replace_cols:
            merged.pop(col)
            
        merged[col_new_name] = merged_col[lookup.name]
    
    return merged

def file_parts(filepath):
    inf = 1000
    exts = []
    for _ in range(inf):
        suffix = filepath.suffix[1:]
        if suffix not in {'data', 'ent', 'be', 'txt', 'csv', 'zip', 'meta', 'json'}:
            break
        exts.append(suffix)
        new_filepath = filepath.with_suffix('')
        if new_filepath == filepath:
            break
        filepath = new_filepath
    
    return filepath.name, '.' + '.'.join(exts[::-1])

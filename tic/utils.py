from itertools import islice

def windowed(seq, window_size=0, step=1):
    for i in range(window_size, len(seq)-window_size, step):
        s = slice(i-window_size, i+window_size+1)
        yield seq[s]

def write_tokens(tokens, filename, sep, window_size=0):
    assert len(tokens) * (2*window_size+1) < 500*2**24
    processed = "\n".join(sep.join(ts) for ts in windowed(tokens, window_size))

    with open(filename, 'w') as fw:
        fw.write(processed)

def read_tokens(filename, sep):
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            toks = (t.strip() for t in line.split(sep))
            toks = [t for t in toks if t]
            if toks:
                yield toks
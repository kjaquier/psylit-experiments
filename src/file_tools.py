import re
import pathlib
from utils.misc import FuncRegister

COMMANDS = FuncRegister()

VERBOSE_MODE = False


def _log(*args, **kwargs):
    if VERBOSE_MODE:
        return print(*args, **kwargs)
    return None


# def file_parts_old(filename):
#     known_extensions = {'data', 'ent', 'be', 'txt', 'csv', 'zip', 'meta', 'json'}
#     exts = []
#     base = None
#     sf = filename.split('.')
#     for i, x in enumerate(reversed(sf)):
#         if x not in known_extensions:
#             cut = -1 - i
#             ext = '.'.join(sf[cut:])
#             base = '.'.join(sf[:cut])
#             return base, ext
#     raise Exception(f'Cannot get parts for: {filename}')

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
    
    return filepath.name, '.' + '.'.join(exts)
           

@COMMANDS.register
def add_ext(files, *args, dry_run=False):
    """Usage: <op><ext>[ <op><ext>...]"""
    for arg in args:
        op, ext = arg[0], arg[1:]
        if op == '+':
            
            for f in files:
                new_f = f"{f}{ext}"
                _log(f"'{f}' -> '{new_f}'")
                if not dry_run:
                    f.rename(new_f)

@COMMANDS.register
def substitute_base(files, *args, dry_run=False):
    """Usage: <char> [<char>...] <replacing>"""
    *pats, replacing = args
    #raise Exception(f"{args} => {pats!r}, {replacing!r}")
    regex = re.compile('|'.join(re.escape(p) for p in pats))
    for f in files:
        base, ext = file_parts(f)
        newbase = regex.sub(replacing, base)
        new_f = f.parent / f"{newbase}{ext}"

        if f == new_f:
            continue

        _log(f"'{f}' -> '{new_f}'")
        if not dry_run:
            #print("I WILL DO IT")
            f.rename(new_f)


def main(command: f"Command to run ; available: {'|'.join(COMMANDS.keys())}",
         directory: "Directory to run command on",
         file_pattern: ("Pattern for matching files", 'option', 'p')='**/*',
         dry_run: ("Don't actually rename the files", 'flag', 'n')=False,
         verbose: ("Print previous and new names", 'flag', 'v')=False,
         *args: "Command arguments"):
    
    global VERBOSE_MODE
    VERBOSE_MODE = verbose

    args_list = [a.strip() for a in args]
    cmd = COMMANDS[command]
    d = pathlib.Path(directory)
    files = [f for f in d.glob(file_pattern) if f.is_file()]
    _log(f"{len(files)} matches")
    cmd_res = cmd(files, *args_list, dry_run=dry_run)
    print(cmd_res if cmd_res is not None else 'Done')


if __name__ == '__main__':
    import plac
    plac.call(main)

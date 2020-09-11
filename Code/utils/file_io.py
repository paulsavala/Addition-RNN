import json
import ast
from pathlib import Path


def delete_folder(path):
    if path.exists():
        for sub in path.iterdir():
            if sub.is_dir():
                delete_folder(sub)
            else:
                sub.unlink()
        path.rmdir()


def list_to_csv(l, path, headers=None):
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        open_mode = 'a'
        write_headers = False
    else:
        open_mode = 'w'
        write_headers = True
    with open(path, open_mode) as f:
        if headers and write_headers:
            f.write(headers)
            f.write('\n')
        for x in l:
            f.write(x)
            f.write('\n')


def append_or_write(path, s, newline=False):
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        open_mode = 'a'
    else:
        open_mode = 'w'
    with open(path, open_mode) as f:
        f.write(s)
        if newline:
            f.write('\n')


def smart_load(s, cleanup=False):
    # Takes a string s and tries to figure out what data type it is
    if cleanup:
        s = s.strip('\n').strip()
    try:
        # dict?
        return json.loads(s)
    except:
        pass
    if s.isdigit():
        # int?
        try:
            return int(s)
        except:
            pass
    if s.isnumeric():
        # float?
        try:
            return float(s)
        except:
            pass
    if s.isidentifier():
        # reserved keyword (like None or def)?
        try:
            s = ast.literal_eval(s)
            return s
        except:
            pass
    if s[0] == '[' and s[-1] == ']':
        # a list?
        try:
            s = ast.literal_eval(s)
            return s
        except:
            pass
    # it's a string
    return s
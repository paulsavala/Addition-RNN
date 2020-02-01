from pathlib import Path


def delete_folder(path):
    if path.exists():
        for sub in path.iterdir():
            if sub.is_dir():
                delete_folder(sub)
            else:
                sub.unlink()
        path.rmdir()


def append_or_write(path, s):
    if path.exists():
        open_mode = 'a'
    else:
        open_mode = 'w'
    with open(path, open_mode) as f:
        f.write(s)
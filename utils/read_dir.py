import os


def get_file_from_dir(dir_name, ext='.arrow'):
    files = []
    for parent, dirnames, filenames in os.walk(dir_name):
        for filename in filenames:
            if filename.lower().endswith(
                    (ext)):
                files.append(os.path.join(parent, filename))
    return files

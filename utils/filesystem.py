import os
import fnmatch


def find_files(directory, pattern, path=True):
    '''Recursively finds all files matching the pattern.'''

    files = []
    for root, dirnames, filenames in os.walk(
        directory, followlinks=True):
        for filename in fnmatch.filter(filenames, pattern):
            if path:
                files.append(os.path.join(root, filename))
            else:
                files.append(filename)
    return files

from glob import glob

use_files = ["../requirements.txt", "../test-requirements.txt"]
files = set()

# Collect a list of dependencies
for ifile in use_files:
    with open(ifile, 'r') as ff:
        lines = ff.readlines()
        files = files.union(lines)

# Sort and save
files = list(files)
files.sort()
with open('./requirements.txt', 'w') as ff:
    ff.writelines(files)

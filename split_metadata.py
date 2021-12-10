# Split ECCO metadata file into train, dev, and test.

import csv
import argparse
import random
import math
import os
import pathlib
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('csv', help="A CSV file with a header line.")
parser.add_argument('src_dir', help="Directory with source documents, possibly in subdirectories.")
parser.add_argument('out_dir', help="Directory to which train, dev, and test CSV files are saved.")
args = parser.parse_args()

with open(args.csv, newline='') as f:
    reader = csv.reader(f, quotechar='"')
    header = next(reader)
    lines = list(reader)
    random.shuffle(lines)

intervals = [math.floor(0.01*len(lines)), math.floor(0.02*len(lines))]

train = lines[intervals[1]:]
dev = lines[intervals[0]:intervals[1]]
test = lines[:intervals[0]]

print(f"{len(train)} train files, {len(dev)} dev files, {len(test)} test files")

# print(glob.glob(args.src_dir + '/' + '0002200101', recursive=True))

# Make a dictionary with document IDs as keys and paths to the source files as values.
# Files that aren't source files are also included, which shouldn't matter.
files = {}
for root, _, fs in os.walk(args.src_dir):
    for f in fs:
        files[f.rstrip('.txt')] = pathlib.PurePath(root, f)

print(list(files.items())[:10])

for l in train[:10]:
    print(l)
    print(files[l[0]])

# for l in train, find the xml file from path
# add a header to each document
# append the result to a list
# after processing all files, shuffle the list and write to file

def document_gen(doc_headers, path_dict):
    for header in doc_headers:
        with open(path_dict[header[0]]) as f:
            lines = [header]
            for line in f:
                line = line.rstrip('\n')
                if line:
                    lines.append(line)
                else:
                    yield lines
                    lines = [header]

for fname, doc_headers in [('train.csv', train), ('dev.csv', dev), ('test.csv', test)]:
    with open(pathlib.PurePath(args.out_dir, fname), 'w', newline='') as f:
        for doc in document_gen(doc_headers, files):
            # TODO: Read to list and shuffle
            f.write('\t'.join(doc[0]) + '\n')
            for line in doc[1:]:
                f.write(line + '\n')
            f.write('\n')

# for fname, ls in [('train.csv', train), ('dev.csv', dev), ('test.csv', test)]:
#     with open(args.out_dir + '/' + fname, 'w', newline='') as f:
#         writer = csv.writer(f, quotechar='"')
#         writer.writerow(header)
#         writer.writerows(ls)

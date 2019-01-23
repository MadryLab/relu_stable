import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Pass in log to parse')
parser.add_argument('--log_name', dest='log_name', help='you should use the same string as the name of the .mat file')
args = parser.parse_args()

filename = 'verification/logs/{}.csv'.format(args.log_name)

f = open(filename, 'rb+')
i = 0

relus = []
for line in f:
    if i > 0:
        fixed_line = str(line, 'utf-8')
        num_relus = int(fixed_line.split(',')[0])
        relus.append(num_relus)
    i += 1
# Assumes a 3 layer neural network was verified
if len(relus) % 3 != 0:
    relus=relus[:-2]
temp = np.reshape(relus, (-1, 3))
rs = np.mean(temp, axis=0)
print("For file {}, it has {} unstable relus on average".format(filename, np.sum(rs)))
print("Layer 1: {}, Layer 2: {}, Layer 3: {}".format(rs[0], rs[1], rs[2]))

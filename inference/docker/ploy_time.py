import sys
import numpy as np
import matplotlib.pyplot as plt

infile = sys.argv[1]
f = open(infile, 'r')
y = []
for line in f.readlines():
    line = line.strip()
    if not len(line):
        continue;
    value = line.split(' ')
    if len(value) < 1:
        continue
    t = float(value[1].strip())
    y.append(t)

#y = [float(line.split(' ')[1].strip()) for line in f.readlines()]


x = range(0, len(y))
plt.figure()
plt.plot(x,y)
plt.xlabel("Samples")
plt.ylabel("Latency (Sec)")
plt.show()

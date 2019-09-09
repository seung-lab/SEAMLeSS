import sys

infile = sys.argv[1]

#print("args 0 is ", infile)

f = open(infile, 'r')

n = 0
total = 0.0
max_t = 0
min_t = 1000

wtotal = 0.0
wmax_t = 0
wmin_t = 1000
for line in f.readlines():
    line = line.strip()
    if not len(line):
        continue;
    value = line.split(' ')
    if len(value) < 1:
        continue
    t = float(value[1].strip())
    max_t = max(t, max_t)
    min_t = min(min_t, t)
    total += t
    n+=1;
    if len(value)==4:
        t = float(value[3].strip())
        wmax_t = max(t, wmax_t)
        wmin_t = min(wmin_t, t)
        wtotal += t


print("avg", total/n, "max", max_t, "min", min_t)
print("write avg", wtotal/n, "max", wmax_t, "min", wmin_t)


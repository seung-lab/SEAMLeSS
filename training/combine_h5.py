import h5py
import sys

outname = sys.argv[1]
fnames = sys.argv[2:]

outf = h5py.File('{}.h5'.format(outname), 'w')
for fname in fnames:
    inf = h5py.File(fname, 'r')
    name = fname[:fname.index('.h5')]
    print('Loading {}...'.format(name))
    outf.create_dataset(name, data=inf['main'][:])
    inf.close()
outf.close()

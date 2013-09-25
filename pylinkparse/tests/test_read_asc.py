import numpy as np

path = 'pylinkparse/tests/data/'
fname = path + 'test_raw.asc'

saccades = []
blinks = []
fixations = []
with open(fname) as fid:
    for line in fid:
        if 'ESACC' in line:
            saccades.append(line)
        elif 'EBLINK' in line:
            blinks.append(line)
        elif 'EFIX' in line:
            fixations.append(line)

saccades = np.genfromtxt(saccades, dtype=(['S8'] * 2) + (['f8'] * 11))

blinks = np.genfromtxt(blinks, dtype=(['S8'] * 2) + (['f8'] * 3))

fixations = np.genfromtxt(fixations, dtype=(['S8'] * 2) + (['f8'] * 8))

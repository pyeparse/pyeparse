# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import pandas as pd
from sklearn.cross_validation import Parallel, delayed
import os
import os.path as op
# from parse import data_frame_from_log
import pyximport
pyximport.install()
from parsextended import data_frame_from_log

paths = ['exp1/data/series_4', 'exp2/data']

logs = [[op.join(path, f) for f in os.listdir(path) if (f.endswith('.asc')
        and len([c for c in f if c.isdigit()]) == 6)] for path in
        paths]

out_names = ['OSSRC_final_face_object.csv',
             'OSSRC_final_face_hand.csv']
n_trials = [384, 192]

for logfiles, outname, n_trials_ in zip(logs, out_names, n_trials):
    my_dfs = Parallel(n_jobs=8)(delayed(data_frame_from_log)
                                (l, n_trials_) for l in logfiles)
    pd.concat(my_dfs).to_csv(outname)

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)


import pandas as pd
from itertools import count
import numpy as np
cimport cython

cdef: 
    char *CODE_SAC = 'ESACC'
    char *CODE_FIX = 'EFIX'
    char *CODE_BLINK = 'EBLINK'
    dict GAZE_DATA = dict(zip(['eye', 'stime', 'etime', 'dur', 'sxp', 'syp',
                               'exp', 'eypampl', 'axp', 'ayp', 'aps',
                               'latency'],
                               [''] + [0.0] * 11))

    struct _GAZE_DATA:
        char *eye 
        double stime
        double etime 
        double dur 
        double sxp
        double syp
        double exp
        double eypampl
        double axp
        double ayp
        double aps

    struct _SAMPLE:
        double time
        double xpos
        double ypos
        double ps
        double xvel
        double yvel
        double xref
        double yref

    struct _MESSAGE:
        double start_time
        unsigned int trial_no
        char *mark
        char *condition
        unsigned int block
        char *expected
        double prestime

    struct _BUTTON:
        double time 
        unsigned int trial
        unsigned int responses
        unsigned int button
        char *type
        char *code
        unsigned int unc_dms


    cdef struct _MODALITY:
        double time
        char *modality
        char *mark


    _GAZE_DATA _FIXATION
    _GAZE_DATA _SACCADE
    _GAZE_DATA _BLINK

_BLINK.sxp = 0.0
_BLINK.syp = 0.0
_BLINK.exp = 0.0
_BLINK.eypampl = 0.0
_BLINK.axp = 0.0
_BLINK.ayp = 0.0
_BLINK.aps = 0.0
_SACCADE.axp = 0.0
_SACCADE.ayp = 0.0
_SACCADE.aps = 0.0
_FIXATION.sxp = 0.0
_FIXATION.exp = 0.0
_FIXATION.syp = 0.0
_FIXATION.eypampl = 0.0
 
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _parse_fixation(char *line):
    cdef:
        list split
    out = _FIXATION
    split = line.strip('\n\t').split()[1:]
    out.eye = split[0]
    out.stime = float(split[1])
    out.etime = float(split[2])
    out.dur = float(split[3])
    out.axp = float(split[4])
    out.ayp = float(split[5])
    out.aps = float(split[6])

    return out

def parse_fixation(char *line):
    return _parse_fixation(line)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _parse_saccades(char *line):
    cdef:
        list split

    out = _SACCADE
    split = line.strip('\n\t').split()[1:]
    out.eye = split[0]
    out.stime = float(split[1])
    out.etime = float(split[2])
    out.dur = float(split[3])
    out.sxp = float(split[4])
    out.syp = float(split[5])
    out.exp = float(split[6])
    out.eypampl = float(split[7])

    return out

def parse_saccades(char *line):
    return _parse_saccades(line)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _parse_blink(char *line):
    cdef:
        list split
   
    out = _BLINK
    split = line.strip('\n\t').split()[1:]
    out.eye = split[0]
    out.stime = float(split[1])
    out.etime = float(split[2])
    out.dur = float(split[3])
   
    return out

def parse_blink(char *line):
    return _parse_blink(line)

# ctypedef _SAMPLE SAMPLE
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _parse_samples(char *line):
    cdef: 
        list split
        _SAMPLE out

    split = line.strip('\n\t').split()[1:]
    out.time = float(split[0])
    out.xpos = float(split[1])
    out.ypos = float(split[2])
    out.ps = float(split[3])
    out.xvel = float(split[4])
    out.yvel = float(split[5])
    out.xref = float(split[6])
    out.yref = float(split[7])
    return out

def parse_samples(char *line):
    return _parse_samples(line)


@cython.boundscheck(False)
cdef _parse_message(char *line):
    cdef: 
        list split
        list sublist
        _MESSAGE out
    tmp = line[4:].replace(' ', '_')
    line = tmp
    split = line.split('_')
    sublist = split[-1].split('-')
    out.start_time = float(split[0])
    out.trial_no = int(split[1][6:])
    tmp = split[2]
    out.mark = tmp
    tmp = split[3]
    out.condition = tmp
    out.block = int(split[4][6:])
    out.expected = sublist[1]
    out.prestime = float(sublist[3])
    return (out.start_time, out.trial_no, out.mark, out.condition, out.block,
            out.expected, out.prestime)

def parse_message(char *line):
    return _parse_message(line)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _parse_button(char *line):
    cdef:
        list split
        list other
        _BUTTON out

    split = line.strip('MSG ').split()[1:]
    out.trial = int(''.join([c for c in split[0] if c.isdigit()]))
    other = [c.split(':')[1] for c in split[1:]]
    out.responses = int(other[0])
    out.button = int(other[1])
    out.type = other[2]
    out.code = other[3]
    out.time = float(other[4])
    out.unc_dms = int(other[5])
    return  out

def parse_button_press(char *line):
    return _parse_button(line)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _parse_modality(char *line):
    cdef:
        list split 
        _MODALITY out
    split = line.strip('\n\t').split()[1:]
    tmp1, tmp2 = split[1].split('-')[1:]
    out.time = float(split[0])
    out.modality = tmp1
    out.mark = tmp2
    return out

def parse_modality(char *line):
    return _parse_modality(line)

def is_start_trial(char *line):
    cdef:
        unsigned int out = 0
        char *term1 = 'MSG'
        char *term2 = 'trial'
        char *term3 = 'start'

    if term1 in line and term2 in line and term3 in line:
        out = 1
    return out


def is_manual_response(char *line):
    cdef:
        unsigned int out = 0
        char *sep = ':'
        char *term = 'responses'
    if sep in line and term in line: 
        out = 1
    return out 

trial_fields = ['subject', 'idx', 'modality', 'condition', 'block',
                'trial_no', 'manual_responses', 'n_manual_response',
                'manual_latencies', 'expected',
                'prestime']


def iter_modality(object fid, char *start_block_tag='BLOCK-START',
                  char *stop_block_tag='BLOCK-END'):
    cdef:
        unsigned int do_read = 0
        char *modality
        char *mod = 'Modality'
        char *start = 'START'

    for line in fid:
        if start_block_tag in line:
            do_read = 1
        elif stop_block_tag in line:
            do_read = 0
        elif mod in line and start in line:
            """ MODALITY SWITCH
            """
            p = parse_modality(line)['modality']
            modality = p
        if do_read:
            yield modality, line

@cython.boundscheck(False)
@cython.wraparound(False)
def merge_log_records(dict trial,unsigned int cnt,
                      missing=np.nan):
    """ Merge hand and eye data as parsed by iter trials.

    Duplicate the manual response entries such
    that you get something like (cartoon example):

    trial | mresp | mlat | saccade_resp | saccade_lat

    1       l       350    l              345
    1       l       350    r              237
    2       3       456    r              267
    2       3       456    l              407

    Parameters
    ----------
    trial : dict
        The trial raw data as initially parsed from the ASCII EyeLink
        file.
    Returns
    -------
    df : instance of pandas.core.frame.DataFrame
        The manual and ocular responses combined in one data frame
        representing the events of interest for a given trial
    """
    cdef:
        list eye_tracking = ['saccades', 'fixations', 'blinks']
        list final = []
        list dfs = []
        char *a, *b
        str key, k
        dict general, this, rec, d
 
    general = {k: v for k, v in trial.items() if k not in eye_tracking}
    a, b = 'manual_latencies', 'manual_responses'
    this = general.copy()
    if general[a] != []:
        for lat, resp in zip(general[a], general[b]):
            this.update({a: lat, b: resp})
    else:  # set error codes
        this.update({a: missing, b: missing})
    final += [this]

    # merge manual responses into eye tracking records
    # and assemble final structure
    for rec in final:
        for key in eye_tracking:
            rec.update({'idx': cnt, 'event_type': key})
            [d.update(rec) for d in trial[key]]
            dfs.extend(trial[key])

    return dfs

@cython.boundscheck(False)
@cython.wraparound(False)
def pick_responses(object df, list trial_fields,
                   unsigned int threshold=200):
    """Reduce data frame records to single lines

    The first saccades in the expected direction exceeding
    a certain threshold (start-position - end-position)
    will be included.
    This is simplification is conservative because bad
    classifications, e.g., earlier responses missed wrong directions,
    would work against the hypotheses pursued.
    """
    cdef:
        double target_latency
        unsigned int no_resp

    if len(df) > 3:  # multple responses
        saccades = df[df.event_type == 'saccades']
        target_saccade = None

        # find the correct saccade
        if saccades.expected.unique()[0] == 'r':
            idx = (saccades.sxp - saccades.exp < -threshold).nonzero()[0]
            if len(idx):
                target_saccade = saccades.to_records()[idx][:1]
        if saccades.expected.unique()[0] == 'l':
            idx = (saccades.sxp - saccades.exp > threshold).nonzero()[0]
            if len(idx):
                target_saccade = saccades.to_records()[idx][:1]

        if target_saccade is not None:
            target_latency = target_saccade['latency'][0]
            no_resp = 0
        else:
            target_latency = 0.
            target_saccade = saccades.to_records()[:1]
            no_resp = 1

        # find fixations following saccades
        fixations = df[df.event_type == 'fixations']
        k = fixations.latency > target_latency
        fixation = fixations[k].to_records()[:1]

        # find blinks
        blink = df[df.event_type == 'blinks'].to_records()[:1]
        df = pd.DataFrame(np.r_[target_saccade, fixation, blink])
        df['gaze_resp'] = 'missing' if no_resp else 'correct'
    else:
        # eye only, hands treated separately.
        df['gaze_resp'] = 'missing'  # becomes df.saccade_resp

    # assemble and populate output table
    out = pd.DataFrame(df[trial_fields].to_records()[:1])
    if ('saccades' in df.event_type.tolist() and
            (df['gaze_resp'] != 'missing').any()):
        out['saccade_lat'] = target_latency
        out['saccade_resp'] = df.gaze_resp.values[0]
    else:
        out['saccade_lat'], out['saccade_resp'] = [np.nan] * 2

    if 'fixations' in df.event_type:
        fixations = df[df.event_type == 'fixations']
        out['fixation_duration'] = fixations.dur.values[0]
    else:
        out['fixation_duration'] = np.nan

    if 'blinks' in df.event_type:
        out['blinks'] = len(df[df.event_type == 'blinks'])
    else:
        out['blinks'] = np.nan

    return out

@cython.boundscheck(False)
def data_frame_from_log(char *fname, unsigned int n_trials,
                        unsigned int time_out=1600):
    """ Assemble data table from log file
    Parameters
    ----------
    fname : str
        The name of the logfile
    n_trials :  int
        the number of trials expected
    time_out : int
        When to stop recording data for a given trial
    """
    sub = ''.join([k for k in fname.split('/')[-1] if k.isdigit()])
    cdef:
        object trial_count = count(1)
        object index_count = count(0)
        object fid = open(fname)
        unsigned int this_count = 0
        unsigned int do_listen = 0
        unsigned int this_trial = 0 
        unsigned int trial_no, n_manual_response
        list trials = []
        list dfs = []
        list saccades
        list fixations
        list blinks
        list default
        list manual_responses
        list manual_latencies
        str expected 
        str subject = sub
        str line, modality
        double time, lat, start_time
        dict trial = {}

    for modality, line in iter_modality(fid):
        if is_start_trial(line):
            """ TRIAL START
            """
            this_count = trial_count.next()
            trial_no = this_count
            n_manual_response = 0
            saccades = []
            fixations = []
            blinks = []
            manual_responses = []
            manual_latencies = []
            start_time, trial_no, mark, condition, block, \
                expected, prestime = parse_message(line)
            do_listen = 1

        elif is_manual_response(line) and do_listen:
            button = parse_button_press(line)
            # print line
            if expected in 'l':
                manual_responses += ([1] if button['button'] == 1
                                     else [0])
            if expected in 'r':
                manual_responses += ([1] if button['button'] == 2
                                     else [0])

            lat = button['time'] - prestime
            manual_latencies += [lat] if lat < time_out else [0]
            n_manual_response += button['responses']

        elif CODE_SAC in line and do_listen:  # parse saccade
            """ SACCADE MSG
            """
            saccades.append(parse_saccades(line))
            time = saccades[-1]['stime']
            saccades[-1]['latency'] = time - start_time

        elif CODE_FIX in line and do_listen:  # parse saccade
            """ FIXATION MSG
            """
            fixations.append(parse_fixation(line))
            time = fixations[-1]['stime']
            fixations[-1]['latency'] = time - start_time

        elif CODE_BLINK in line and do_listen:  # parse saccade
            """ BLINK MSG
            """
            blinks.append(parse_blink(line))
            time = blinks[-1]['stime']
            blinks[-1]['latency'] = time - start_time

        elif line[0].isdigit() and do_listen:
            """ TIME TRACKER
            """
            time = float(line[:9])
            lat = time - start_time
            if lat > time_out:
                do_listen = 0
                for k in [fixations, saccades, blinks]:
                    if not k:
                        k.append(GAZE_DATA)
                
                trial.update({'trial_no': trial_no,
                              'n_manual_response': n_manual_response,
                              'saccades': saccades,
                              'fixations': fixations,
                              'blinks': blinks,
                              'modality': modality,
                              'manual_responses': manual_responses,
                              'manual_latencies': manual_latencies,
                              'start_time': start_time,
                              'mark': mark,
                              'condition': condition,
                              'block': block,
                              'expected': expected,
                              'prestime': prestime,
                              'subject': subject})
                df = merge_log_records(trial, index_count.next())
                dfs.append(pick_responses(pd.DataFrame(df).dropna(0, 'all'),
                                          trial_fields))
                trials.append(trial)
    fid.close()

    assert len(trials) == n_trials
    return pd.concat(dfs)

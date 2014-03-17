# -*- coding: utf-8 -*-
"""
Example usage (and testing) of the edf2py ctypes wrapper for the EDF Access API

This example is a work in progress, and is intended to test each function
available from edf2py. By creating this example, it also becomes clear
where higher level python functions may be of use to hide some of the details
when using the edfapi ctypes wrapper directly.

NOTE: The example has a hard coded edf file name and trial start and end string
      defines. At minimum the edf_name must be changed to a relative or abs
      path to a valid EDF file.

TO RUN:

    python example.py

todo:
   - Add usage of the various trial level grouping functions
   - Add usage of the EDF file bookmark related fucntions
   - ???

Created on Sat Mar 15 09:40:17 2014

@author: Sol
"""

from os import path as op
from pyeparse.edf._edf2py import (get_revision, get_eyelink_revision,
                                  set_trial_identifier, get_trial_count,
                                  get_element_count, get_next_data)
from pyeparse.edf._defines import event_constants
from pyeparse.edf._handlers import (element_handlers, default_handler,
                                    edf_open, preamble_text)


fname = op.join(op.dirname(__file__), "../pyeparse/tests/data/test_2_raw.edf")
# See comments for edf_set_trial_identifier func in edf.h
#
trial_start_str = "TRIALID"
trial_end_str = None


with edf_open(fname) as edf:
    # Open the EDF file for processing
    if edf is None:
        raise RuntimeError("Error opening '%s'. Exiting Demo." % fname)
    print(preamble_text(edf))
    print('** EDF Revision: %s' % get_revision(edf))
    print('** EyeLink Model: %s', get_eyelink_revision(edf))

    sti_result = set_trial_identifier(edf, trial_start_str,
                                      trial_end_str)
    num_trials = get_trial_count(edf)
    print("** Number Trials in Recording: %s" % num_trials)
    item_count = get_element_count(edf)
    print("EDF File Element Count: %s" % item_count)
    etype = None
    while etype != event_constants.get('NO_PENDING_ITEMS'):
        etype = get_next_data(edf)
        r = element_handlers.get(event_constants[etype],
                                 default_handler)(etype, edf)
        if r:
            if etype == event_constants['MESSAGEEVENT']:
                print(r)
            else:
                print(r['name'], r.get('sttime', ''))

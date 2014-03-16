# -*- coding: utf-8 -*-
"""
Example usage (and testing) of the edf2py ctypes wrapper for the EDF Access API.

This example is a work in progress, and is intended to test each function
available from edf2py. By creating this example, it also becomes clear
where higher level python functions may be of use to hide some of the details 
when using the edfapi ctypes wrapper directly.

NOTE: The example has a hard coded edf file name and trial start and end string
      defines. At minimum the edf_name must be changed to a relative or abs
      path to a valid EDF file.

TO RUN:

    python example.py
      
TODO:
   - Add usage of the various trial level grouping functions
   - Add usage of the EDF file bookmark related fucntions
   - ???

Created on Sat Mar 15 09:40:17 2014

@author: Sol      
"""
import edf2py
import os, sys
import ctypes as ct

file_name = "../../tests/data/test_2_raw.edf"
# See comments for edf_set_trial_identifier func in edf.h
# 
trial_start_str="TRIALID"
trial_end_str=None

if __name__ == '__main__':
    from handlers import element_handlers, default_handler
    from edf2py import event_constants
    try:
        # Open the EDF file for processing
        edf = edf2py.edf_file(file_name)
        if edf is None:
            print "Error openning '{0}'. Exiting Demo.".format(file_name)
            sys.exit(0)

        # Get the file preamble
        preambletxt = edf2py.preamble_text(edf)
        print preambletxt

        edf_revision = edf2py.get_revision(edf)
        print '** EDF Revision:', edf_revision

        eyelink_model = edf2py.get_eyelink_revision(edf)
        print '** EyeLink Model:', eyelink_model

        sti_result = edf2py.set_trial_identifier(edf, trial_start_str,
                                                 trial_end_str)
        num_trials = edf2py.get_trial_count(edf)
        print "** Number Trials in Recording:", num_trials

        item_count = edf2py.get_element_count(edf)
        print "EDF File Element Count:", item_count
        print

        etype = None
        while etype != event_constants.get('NO_PENDING_ITEMS'):
            etype = edf2py.get_next_data(edf)
            r = element_handlers.get(event_constants[etype],
                                     default_handler)(etype, edf)
            if r:
                if etype == event_constants.get('MESSAGEEVENT'):
                    print r
                else:
                    print r['name'], r.get('sttime', '')

    except Exception, e:
        import traceback
        traceback.print_exc()
        try:
            edf2py.close_file(edf)
        except:
            pass
        raise e

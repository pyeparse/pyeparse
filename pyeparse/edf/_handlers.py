# -*- coding: utf-8 -*-
"""
handlers.py contains the functions that are called for each element type
within an edf file.

The handlers are basically stubs right now, but the idea
would be that each would get the element data, do whatever processing is needed
on it, and return an the element in the form that will be used by the parent
module.

Created on Sat Mar 15 20:40:48 2014

@author: Sol
"""
from . import _edf2py as edf2py
from ._edf2py import event_constants
from . import _defines as defines
import ctypes as ct

#
## Helper functions
#


def todo(func):
    """
    @ to make it easy to flag handlers that are not yet completed (which is
    all of them right now).
    """

    def wrapper(*args, **kwargs):
        print "** todo: {0}({1},{2})".format(func.__name__, *args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def to_dict(element):
    """
    Accepts any EDF file element type and returns a dict containing all
    'public' fields of the element struct. Fields that are ctypes arrays
    are converted to a python list.
    """
    element_dict = {}
    element_keys = [a for a in dir(element) if not a.startswith('_')]
    for k in element_keys:
        v = getattr(element, k)
        if hasattr(v, '_length_'):
            v = ', '.join(['{0}'.format(i) for i in v])
        element_dict[k] = v
    return element_dict


def sample_fields_available(sflags):
    """
    Returns a dict where the keys indicate fields (or field groups) of a
    sample; the value for each indicates if the field has been populated
    with data and can be considered as useful information.
    """
    return dict(
        # left eye data available
        left=(sflags & defines.SAMPLE_LEFT) != 0,
        # right eye data available
        right=(sflags & defines.SAMPLE_RIGHT) != 0,
        # sample time available
        time=(sflags & defines.SAMPLE_TIMESTAMP) != 0,
        # raw eye position available
        pupilxy=(sflags & defines.SAMPLE_PUPILXY) != 0,
        # href eye position available
        hrefxy=(sflags & defines.SAMPLE_HREFXY) != 0,
        # gaze position available
        gazexy=(sflags & defines.SAMPLE_GAZEXY) != 0,
        # x,y pixels per degree available
        gazeres=(sflags & defines.SAMPLE_GAZERES) != 0,
        # pupil size available
        pupilsize=(sflags & defines.SAMPLE_PUPILSIZE) != 0,
        # sample status field available
        status=(sflags & defines.SAMPLE_STATUS) != 0,
        # sample inputs field available
        inputs=(sflags & defines.SAMPLE_INPUTS) != 0,
        # sample buttons field available
        button=(sflags & defines.SAMPLE_BUTTONS) != 0,
        # sample head position field available
        headpos=(sflags & defines.SAMPLE_HEADPOS) != 0,
        # if this flag is set for the sample add .5ms to the sample time
        addoffset=(sflags & defines.SAMPLE_ADD_OFFSET) != 0,
        # reserved variable-length tagged
        tagged=(sflags & defines.SAMPLE_TAGGED) != 0,
        # user-defineabe variable-length tagged
        utagged=(sflags & defines.SAMPLE_UTAGGED) != 0,
    )


def event_fields_available(eflags):
    """
    Returns a dict where the keys indicate fields (or field groups) of an
    EDF event; the value for each indicates if the field has been populated
    with data and can be considered as useful information.
    """
    return dict(
        # end time (start time always read)
        endtime=(eflags & defines.READ_ENDTIME) != 0,
        # gaze resolution xy
        gres=(eflags & defines.READ_GRES) != 0,
        # pupil size
        size=(eflags & defines.READ_SIZE) != 0,
        # velocity (avg, peak)
        vel=(eflags & defines.READ_VEL) != 0,
        # status (error word)
        status=(eflags & defines.READ_STATUS) != 0,
        # event has start data for vel,size,gres
        beg=(eflags & defines.READ_BEG) != 0,
        # event has end data for vel,size,gres
        end=(eflags & defines.READ_END) != 0,
        # event has avg pupil size, velocity
        avg=(eflags & defines.READ_AVG) != 0,
        # position eye data
        pupilxy=(eflags & defines.READ_PUPILXY) != 0,
        hrefxy=(eflags & defines.READ_HREFXY) != 0,
        gazexy=(eflags & defines.READ_GAZEXY) != 0,
        begpos=(eflags & defines.READ_BEGPOS) != 0,
        endpos=(eflags & defines.READ_ENDPOS) != 0,
        avgpos=(eflags & defines.READ_AVGPOS) != 0,
    )


#
## EDF File Handlers
#

def default_handler(event_type, edf_ptr):
    """
    Handler called if the element type has not been mapped to a specific
    handler func for the given element type. The element type -> handler
    func association is maintained in the element_handlers var in this
    module.
    """
    element = edf2py.get_float_data(edf_ptr)
    # todo: WHat to do if unhandled edf file element is found?
    print "\n!! Warning: Unhandled EDF element type:", event_constants[
        event_type], element


@todo
def handle_recording_info(event_type, edf_ptr):
    """
    Handler for RECORDING_INFO structs.
    todo: Need to parse struct fields for sampling rate, etc.
    """
    e = edf2py.get_recording_data(edf_ptr).contents
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    edict = to_dict(e)
    edict['name'] = ets
    return edict


def handle_message(event_type, edf_ptr):
    """
    Handler func for a MSG event.
    """
    # todo: getting msg text should be this hard, look into how to access str
    # properly.
    msg = edf2py.get_event_data(edf_ptr).contents
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    return dict(name=ets,
                time=msg.sttime,
                text=ct.string_at(ct.byref(msg.message[0]),
                                  msg.message.contents.len + 1)[2:]
                )


# Sample handler does not return anything only because the current
# example prints whatever each element handler returns, and printing
# thousands of sample dicts is a little much. ;)
#
#@todo
def handle_sample(event_type, edf_ptr):
    e = edf2py.get_sample_data(edf_ptr).contents
    # use sflags to know what fields are populated for the given sample
    #sflags = sample_fields_available(e.flags) XXX ?
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    edict = to_dict(e)
    edict['name'] = ets
    #return edict


@todo
def handle_start_blink(event_type, edf_ptr):
    e = edf2py.get_event_data(edf_ptr).contents
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    #eread=event_fields_available(e.read)
    edict = to_dict(e)
    edict['name'] = ets
    return edict


@todo
def handle_end_blink(event_type, edf_ptr):
    e = edf2py.get_event_data(edf_ptr).contents
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    #eread=event_fields_available(e.read)
    edict = to_dict(e)
    edict['name'] = ets
    return edict


@todo
def handle_start_saccade(event_type, edf_ptr):
    e = edf2py.get_event_data(edf_ptr).contents
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    #eread=event_fields_available(e.read)
    edict = to_dict(e)
    edict['name'] = ets
    return edict


@todo
def handle_end_saccade(event_type, edf_ptr):
    e = edf2py.get_event_data(edf_ptr).contents
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    #eread=event_fields_available(e.read)
    edict = to_dict(e)
    edict['name'] = ets
    return edict


@todo
def handle_start_fixation(event_type, edf_ptr):
    e = edf2py.get_event_data(edf_ptr).contents
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    #eread=event_fields_available(e.read)
    edict = to_dict(e)
    edict['name'] = ets
    return edict


@todo
def handle_end_fixation(event_type, edf_ptr):
    e = edf2py.get_event_data(edf_ptr).contents
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    #eread=event_fields_available(e.read)
    edict = to_dict(e)
    edict['name'] = ets
    return edict


@todo
def handle_fixation_update(event_type, edf_ptr):
    e = edf2py.get_event_data(edf_ptr).contents
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    #eread=event_fields_available(e.read)
    edict = to_dict(e)
    edict['name'] = ets
    return edict


@todo
def handle_button(event_type, edf_ptr):
    e = edf2py.get_event_data(edf_ptr).contents
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    #eread=event_fields_available(e.read)
    edict = to_dict(e)
    edict['name'] = ets
    return edict


@todo
def handle_input(event_type, edf_ptr):
    e = edf2py.get_event_data(edf_ptr).contents
    ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    #eread=event_fields_available(e.read)
    edict = to_dict(e)
    edict['name'] = ets
    return edict


@todo
def handle_start_parse(event_type, edf_ptr):
    #e = edf2py.get_float_data(edf_ptr).contents
    #ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    # TODO: ???
    pass


@todo
def handle_end_parse(event_type, edf_ptr):
    #e = edf2py.get_float_data(edf_ptr).contents
    #ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    # TODO: ???
    pass


@todo
def handle_break_parse(event_type, edf_ptr):
    #e = edf2py.get_float_data(edf_ptr).contents
    #ets = event_constants.get(event_type, "UNKNOWN EDF EVENT TYPE")
    # TODO: ???
    pass


def handle_eof(event_type, edf_ptr):
    """
    Handle end of EDF file condition.
    """
    edf2py.close_file(edf_ptr)

# element_handlers maps the various EDF file element types to the
# element handler function that should be called.
#
element_handlers = dict(RECORDING_INFO=handle_recording_info,
                        STARTBLINK=handle_start_blink,
                        ENDBLINK=handle_end_blink,
                        STARTSACC=handle_start_saccade,
                        ENDSACC=handle_end_saccade,
                        STARTFIX=handle_start_fixation,
                        ENDFIX=handle_end_fixation,
                        FIXUPDATE=handle_fixation_update,
                        MESSAGEEVENT=handle_message,
                        BUTTONEVENT=handle_button,
                        INPUTEVENT=handle_input,
                        SAMPLE_TYPE=handle_sample,
                        STARTPARSE=handle_start_parse,
                        ENDPARSE=handle_end_parse,
                        BREAKPARSE=handle_break_parse,
                        NO_PENDING_ITEMS=handle_eof)

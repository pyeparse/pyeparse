/*****

NOTE: This file is actually a combination of the edf.h, edf_data.h, and 
edftypes.h files. 

Content has also been modified a bit to make ctypesgen happy.


*******/

/***********************************************************************
 *
 * EYELINK PORTABLE EXPT SUPPORT
 * Copyright (c) 1996-2002 SR Research Ltd.
 *
 * For non-commercial use only
 *
 * EyeLink library data extensions
 *        VERSION 2.5
 * UPDATED for EyeLink DLL V2.1: added LOST_DATA_EVENT
 * This module is for user applications
 * Use is granted for non-commercial
 * applications by Eyelink licensees only
 *
 * UNDER NO CIRCUMSTANCES SHOULD PARTS OF THESE FILES
 * BE COPIED OR COMBINED.  This will make your code
 * impossible to upgrade to new releases in the future,
 * and SR Research will not give tech support for
 * reorganized code.
 *
 * This file should not be modified. If you must modify it,
 * copy the entire file with a new name, and change the
 * the new file.
 *
 * EDFTYPES.H has platform-portable definitions COPYRIGHT
 * TERMS:  This file is part of the EyeLink application
 * support package.  You may compile this code, and may
 * read it to understand the EyeLink data structures.
 *
 *
 * DO NOT MODIFY THIS CODE IN ANY WAY.
 * SR Research will not support any problems caused by
 * modifying this code, and using modified code will make
 * your programs incompatible with future releases.
 *
 * This code has been tested on many 16 and
 * 32-bit compilers, including Borland and
 * Microsoft, GCC, in Windows, MAC OS X, Linux and DOS.
 * It should  compile without warnings on most platforms
 * If you must port to a new platform, check
 * the data types defined in EDFTYPES.H for
 * correct sizes.
 *
 *
 *
***********************************************************************/

/*!
    file edf.h
 */

/*!
mainpage EyeLink EDF Access API 

section intro Introduction

The EyeLink EDF Access API is a set of C functions that provide access to 
EyeLink EDF files. The access method is similar to that of the online data 
access API, where the program performs a set of eyelink_get_next_data() and 
eyelink_get_float_data() calls to step through the data.

The EDF Access API also provides functions for setting bookmarks within an 
EDF file, and for automatically parsing an EDF file into a set of trials, 
with functions for stepping through the trial set.

As an example use for the API, the edf2asc translator program has been re-written 
to use the API for EDF data access. The source code for this edf2asc program is 
included with the API distribution.

This is the first release of the EDF Access API and should be considered a beta release.

Please report any functionality comments or bugs to support@sr-research.com.

*/



/***********************************************************************
 * additional defines - there are more defines defined in edf_data.h
 ***********************************************************************/
#define NO_PENDING_ITEMS 0
#define RECORDING_INFO 30

#ifdef __cplusplus
extern "C" {
#endif


typedef unsigned char  byte;
typedef short          INT16;
typedef int           INT32;
typedef unsigned short UINT16;
typedef unsigned int  UINT32;

typedef unsigned long long UINT64 ;
typedef long long INT64 ;
typedef unsigned __int64 UINT64 ;
typedef __int64 INT64 ;

typedef struct {
		INT32  msec;	/* SIGNED for offset computations */
		INT16  usec;
} MICRO ;


/*********** EYE DATA FORMATS **********/

/* ALL fields use MISSING_DATA when value was not read, */
/* EXCEPT for <buttons>, <time>, <sttime> and <entime>, which use 0. */
/* This is true for both floating point or integer variables. */

	/* Both samples and events may have             */
	/* several fields that have not been updated.   */
	/* These fields may be detected from the        */
	/* content flags, or by testing the field value */
	/* against these constants: */

#define MISSING_DATA -32768     /* data is missing (integer) */
#define MISSING -32768
#define INaN -32768


	/* binocular data needs to ID the eye for events */
	/* samples need to index the data */
	/* These constants are used as eye identifiers */

#define LEFT_EYE  0   /* index and ID of eyes */
#define RIGHT_EYE 1
#define LEFTEYEI  0
#define RIGHTEYEI 1
#define LEFT      0
#define RIGHT     1

#define BINOCULAR 2   /* data for both eyes available */


/********* EYE SAMPLE DATA FORMATS *******/

	/* The SAMPLE struct contains data from one 4-msec  */
	/* eye-tracker sample. The <flags> field has a bit for each */
	/* type of data in the sample. Fields not read have 0 flag */
	/* bits, and are set to MISSING_DATA */

/* flags to define what data is included in each sample.   */
/* There is one bit for each type.  Total data for samples */
/* in a block is indicated by these bits in the <sam_data> */
/* field of ILINKDATA or EDF_FILE, and is updated by the    */
/* STARTSAMPLES control event. */

#define SAMPLE_LEFT      0x8000  /* data for these eye(s) */
#define SAMPLE_RIGHT     0x4000

#define SAMPLE_TIMESTAMP 0x2000  /* always for link, used to compress files */

#define SAMPLE_PUPILXY   0x1000  /* pupil x,y pair */
#define SAMPLE_HREFXY    0x0800  /* head-referenced x,y pair */
#define SAMPLE_GAZEXY    0x0400  /* gaze x,y pair */
#define SAMPLE_GAZERES   0x0200  /* gaze res (x,y pixels per degree) pair */
#define SAMPLE_PUPILSIZE 0x0100  /* pupil size */
#define SAMPLE_STATUS    0x0080  /* error flags */
#define SAMPLE_INPUTS    0x0040  /* input data port */
#define SAMPLE_BUTTONS   0x0020  /* button state: LSBy state, MSBy changes */

#define SAMPLE_HEADPOS   0x0010  /* * head-position: byte tells # words */
#define SAMPLE_TAGGED    0x0008  /* * reserved variable-length tagged */
#define SAMPLE_UTAGGED   0x0004  /* * user-defineabe variable-length tagged */
#define SAMPLE_ADD_OFFSET 0x0002 /* if this flag is set for the sample add .5ms to the sample time */



typedef struct {
                INT16 len;
                char c;
               } LSTRING;


/*!\page classIntro
 * The EyeLink EDF access API (edfapi.dll) library defines a number of data types that are 
 * used for data reading, found in eye_data.h and edf.h. The useful parts of these structures 
 * are discussed in the following sections. */



/*! The FSAMPLE structure holds information for a sample in the EDF file. 
 * Depending on the recording options set for the recording session, 
 * some of the fields may be empty.*/

typedef struct {
		 UINT32 time;   /*!< time stamp of sample */
		 /*INT16  type; */	/* always SAMPLE_TYPE */ 
		 

		 float  px[2];   /*!< pupil x */
		 float  py[2];   /*!< pupil y */
		 float  hx[2];   /*!< headref x */
		 float  hy[2];   /*!< headref y */
		 float  pa[2]; 	 /*!< pupil size or area */

		 float gx[2];    /*!< screen gaze x */
		 float gy[2];    /*!< screen gaze y */
		 float rx;       /*!< screen pixels per degree */
		 float ry;       /*!< screen pixels per degree */

		 
		 float gxvel[2];  /*!< gaze x velocity */
		 float gyvel[2];  /*!< gaze y velocity */
		 float hxvel[2];  /*!< headref x velocity */
		 float hyvel[2];  /*!< headref y velocity */
		 float rxvel[2];  /*!< raw x velocity */
		 float ryvel[2];  /*!< raw y velocity */

		 float fgxvel[2]; /*!< fast gaze x velocity */
		 float fgyvel[2]; /*!< fast gaze y velocity */
		 float fhxvel[2]; /*!< fast headref x velocity */
		 float fhyvel[2]; /*!< fast headref y velocity */
		 float frxvel[2]; /*!< fast raw x velocity */
		 float fryvel[2]; /*!< fast raw y velocity */

		 INT16  hdata[8];  /*!< head-tracker data (not pre-scaled) */
		 UINT16 flags;     /*!<  flags to indicate contents */
		 
		 //UINT16 status;       /* tracker status flags    */ 
		 UINT16 input;     /*!< extra (input word) */
		 UINT16 buttons;   /*!< button state & changes */

		 INT16  htype;     /*!< head-tracker data type (0=none) */
		 
		 UINT16 errors;    /*!< process error flags */

} FSAMPLE;


/*!The FEVENT structure holds information for an event in the EDF file. 
 * Depending on the recording options set for the recording session and the event type, 
 * some of the fields may be empty.*/

typedef struct  {
		UINT32 time;       /*!< effective time of event */
		INT16  type;       /*!< event type */
		UINT16 read;       /*!< flags which items were included */


		UINT32 sttime;   /*!< start time of the event */
		UINT32 entime;   /*!< end time of the event*/
		float  hstx;     /*!< headref starting points */ 
		float  hsty;     /*!< headref starting points */
		float  gstx;     /*!< gaze starting points */
		float  gsty;     /*!< gaze starting points */
		float  sta;      /*!< pupil size at start */
		float  henx;     /*!<  headref ending points */
		float  heny;     /*!<  headref ending points */
		float  genx;     /*!< gaze ending points */
		float  geny;     /*!< gaze ending points */
		float  ena;       /*!< pupil size at end */
		float  havx;     /*!< headref averages */
		float  havy;     /*!< headref averages */
		float  gavx;     /*!< gaze averages */
		float  gavy;     /*!< gaze averages */
		float  ava;       /*!< average pupil size */
		float  avel;     /*!< accumulated average velocity */
		float  pvel;     /*!< accumulated peak velocity */
		float  svel;     /*!< start velocity */
		float evel;      /*!< end velocity */
		float  supd_x;   /*!< start units-per-degree */
		float  eupd_x;   /*!< end units-per-degree */
		float  supd_y;   /*!< start units-per-degree */
		float  eupd_y;   /*!< end units-per-degree */

		INT16  eye;        /*!< eye: 0=left,1=right */
		UINT16 status;           /*!< error, warning flags */
		UINT16 flags;           /*!< error, warning flags*/
		UINT16 input;
		UINT16 buttons;
		UINT16 parsedby;       /*!< 7 bits of flags: PARSEDBY codes*/
		LSTRING *message;		/*!< any message string*/
		} FEVENT;

 /// @cond REMOVE
typedef struct  {
		UINT32 time;       /* time message logged */
		INT16  type;       /* event type: usually MESSAGEEVENT */

		UINT16 length;	   /* length of message */
		byte   text[260];  /* message contents (max length 255) */
		} IMESSAGE;


/// @cond TEST
typedef struct  {
		UINT32 time;       /* time logged */
		INT16  type;       /* event type: */

		UINT16 data;	   /* coded event data */
		} IOEVENT;

/*! The RECORDINGS structure holds information about a recording block in an EDF file. 
 * A RECORDINGS structure is present at the start of recording and the end of recording. 
 * Conceptually a RECORDINGS structure is similar to the START and END lines inserted in an EyeLink ASC file. 
 * RECORDINGS with a state field = 0 represent the end of a recording block, and contain information regarding 
 * the recording options set before recording was initiated. */


typedef struct
{
	UINT32 time;		/*!< start time or end time*/
	float sample_rate;  /*!< 250 or 500 or 1000*/
	UINT16 eflags;      /*!< to hold extra information about events */
	UINT16 sflags;      /*!< to hold extra information about samples */
	byte state;			/*!< 0 = END, 1=START */
	byte record_type;	/*!< 1 = SAMPLES, 2= EVENTS, 3= SAMPLES and EVENTS*/
	byte pupil_type;	/*!< 0 = AREA, 1 = DIAMETER*/
	byte recording_mode;/*!< 0 = PUPIL, 1 = CR */
	byte filter_type;   /*!< 1,2,3 */	
	byte  pos_type;		/*!<0 = GAZE, 1= HREF, 2 = RAW*/  /*PARSEDBY_GAZE  PARSEDBY_HREF PARSEDBY_PUPIL*/
	byte eye;			/*!< 1=LEFT, 2=RIGHT, 3=LEFT and RIGHT */


}RECORDINGS;

/*!Any one of the above three data types can be read into a buffer of type ALLF_DATA, which is 
 * a union of the event, sample, and recording buffer formats:*/

typedef union {
		FEVENT    fe;
		IMESSAGE  im;
		IOEVENT   io;
		FSAMPLE   fs;
		RECORDINGS rec;
	      } ALLF_DATA ;


/********** SAMPLE, EVENT BUFFER TYPE CODES ***********/

		/* the type code for samples */
#define SAMPLE_TYPE 200




/************* EVENT TYPE CODES ************/

	/* buffer = IEVENT, FEVENT, btype = IEVENT_BUFFER */

#define STARTPARSE 1 	/* these only have time and eye data */
#define ENDPARSE   2
#define BREAKPARSE 10

			/* EYE DATA: contents determined by evt_data */
#define STARTBLINK 3    /* and by "read" data item */
#define ENDBLINK   4    /* all use IEVENT format */
#define STARTSACC  5
#define ENDSACC    6
#define STARTFIX   7
#define ENDFIX     8
#define FIXUPDATE  9

  /* buffer = (none, directly affects state), btype = CONTROL_BUFFER */

			 /* control events: all put data into */
			 /* the EDF_FILE or ILINKDATA status  */
#define STARTSAMPLES 15  /* start of events in block */
#define ENDSAMPLES   16  /* end of samples in block */
#define STARTEVENTS  17  /* start of events in block */
#define ENDEVENTS    18  /* end of events in block */



	/* buffer = IMESSAGE, btype = IMESSAGE_BUFFER */

#define MESSAGEEVENT 24  /* user-definable text or data */


	/* buffer = IOEVENT, btype = IOEVENT_BUFFER */

#define BUTTONEVENT  25  /* button state change */
#define INPUTEVENT   28  /* change of input port */

#define LOST_DATA_EVENT 0x3F   /* NEW: Event flags gap in data stream */

/************* CONSTANTS FOR EVENTS ************/

	/* "read" flag contents in IEVENT */

				/* time data */
#define READ_ENDTIME 0x0040     /* end time (start time always read) */

			    /* non-position eye data: */
#define READ_GRES    0x0200     /* gaze resolution xy */
#define READ_SIZE    0x0080     /* pupil size */
#define READ_VEL     0x0100     /* velocity (avg, peak) */
#define READ_STATUS  0x2000     /* status (error word) */

#define READ_BEG     0x0001     /* event has start data for vel,size,gres */
#define READ_END     0x0002     /* event has end data for vel,size,gres */
#define READ_AVG     0x0004     /* event has avg pupil size, velocity */

			    /* position eye data */
#define READ_PUPILXY 0x0400    /* pupilxy REPLACES gaze, href data if read */
#define READ_HREFXY  0x0800
#define READ_GAZEXY  0x1000

#define READ_BEGPOS  0x0008    /* position data for these parts of event */
#define READ_ENDPOS  0x0010
#define READ_AVGPOS  0x0020


		       /* RAW FILE/LINK CODES: REVERSE IN R/W */
#define FRIGHTEYE_EVENTS  0x8000 /* has right eye events */
#define FLEFTEYE_EVENTS   0x4000 /* has left eye events */

	/* "event_types" flag in ILINKDATA or EDF_FILE */
	/* tells what types of events were written by tracker */

#define LEFTEYE_EVENTS   0x8000 /* has left eye events */
#define RIGHTEYE_EVENTS  0x4000 /* has right eye events */
#define BLINK_EVENTS     0x2000 /* has blink events */
#define FIXATION_EVENTS  0x1000 /* has fixation events */
#define FIXUPDATE_EVENTS 0x0800 /* has fixation updates */
#define SACCADE_EVENTS   0x0400 /* has saccade events */
#define MESSAGE_EVENTS   0x0200 /* has message events */
#define BUTTON_EVENTS    0x0040 /* has button events */
#define INPUT_EVENTS     0x0020 /* has input port events */


	/* "event_data" flags in ILINKDATA or EDF_FILE */
	/* tells what types of data were included in events by tracker */

#define EVENT_VELOCITY  0x8000  /* has velocity data */
#define EVENT_PUPILSIZE 0x4000  /* has pupil size data */
#define EVENT_GAZERES   0x2000  /* has gaze resolution */
#define EVENT_STATUS    0x1000  /* has status flags */

#define EVENT_GAZEXY    0x0400  /* has gaze xy position */
#define EVENT_HREFXY    0x0200  /* has head-ref xy position */
#define EVENT_PUPILXY   0x0100  /* has pupil xy position */

#define FIX_AVG_ONLY    0x0008  /* only avg. data to fixation evts */
#define START_TIME_ONLY 0x0004  /* only start-time in start events */

#define PARSEDBY_GAZE   0x00C0  /* how events were generated */
#define PARSEDBY_HREF   0x0080
#define PARSEDBY_PUPIL  0x0040


/************ STATUS FLAGS (samples and events) ****************/

#define LED_TOP_WARNING       0x0080    /* marker is in border of image*/
#define LED_BOT_WARNING       0x0040
#define LED_LEFT_WARNING      0x0020
#define LED_RIGHT_WARNING     0x0010
#define HEAD_POSITION_WARNING 0x00F0    /* head too far from calibr???*/

#define LED_EXTRA_WARNING     0x0008    /* glitch or extra markers */
#define LED_MISSING_WARNING   0x0004    /* <2 good data points in last 100 msec) */
#define HEAD_VELOCITY_WARNING 0x0001    /* head moving too fast */

#define CALIBRATION_AREA_WARNING 0x0002  /* pupil out of good mapping area */

#define MATH_ERROR_WARNING   0x2000  /* math error in proc. sample */

/* THESE CODES ONLY VSLID FOR EYELINK II */

            /* this sample interpolated to preserve sample rate
               usually because speed dropped due to missing pupil */
#define INTERP_SAMPLE_WARNING 0x1000

            /* pupil interpolated this sample
               usually means pupil loss or
               500 Hz sample with CR but no pupil
            */
#define INTERP_PUPIL_WARNING  0x8000

            /* all CR-related errors */
#define CR_WARNING       0x0F00
#define CR_LEFT_WARNING  0x0500
#define CR_RIGHT_WARNING 0x0A00

            /* CR is actually lost */
#define CR_LOST_WARNING        0x0300
#define CR_LOST_LEFT_WARNING   0x0100
#define CR_LOST_RIGHT_WARNING  0x0200

            /* this sample has interpolated/held CR */
#define CR_RECOV_WARNING       0x0C00
#define CR_RECOV_LEFT_WARNING  0x0400
#define CR_RECOV_RIGHT_WARNING 0x0800





#define TFLAG_MISSING   0x4000    /* missing */
#define TFLAG_ANGLE     0x2000    /* extreme target angle */
#define TFLAG_NEAREYE   0x1000    /* target near eye so windows overlapping */
/*  DISTANCE WARNINGS (limits set by remote_distance_warn_range command)*/
#define TFLAG_CLOSE     0x0800    /* distance vs. limits */
#define TFLAG_FAR       0x0400
/* TARGET TO CAMERA EDGE  (margin set by remote_edge_warn_pixels command)*/
#define TFLAG_T_TSIDE   0x0080    /* target near edge of image (left, right, top, bottom)*/
#define TFLAG_T_BSIDE   0x0040
#define TFLAG_T_LSIDE   0x0020
#define TFLAG_T_RSIDE   0x0010
/* EYE TO CAMERA EDGE  (margin set by remote_edge_warn_pixels command)*/
#define TFLAG_E_TSIDE   0x0008    /* eye near edge of image (left, right, top, bottom) */
#define TFLAG_E_BSIDE   0x0004
#define TFLAG_E_LSIDE   0x0002
#define TFLAG_E_RSIDE   0x0001


/***********************************************************************
 * compile conditions for porting.
 ***********************************************************************/
#define EXPORT
#define DLLExport
#define __stdcall

typedef enum
{
	GAZE,
	HREF,
	RAW
}position_type;



#define PUPIL_ONLY_250 0
#define PUPIL_ONLY_500 1
#define PUPIL_CR	   2


/* @struct TRIAL
     The TRIAL structure is used to access a block of data within an EDF file 
	 that is considered to be a trial within the experimental session. The start time 
	and end time of a TRIAL are defined using the edf_set_trial_identifier() function, 
	where a start and end message text token is specified.

   	@param rec  recording information about the current trial
	@param duration  duration of the current trial
	@param starttime start time of the trial 
	@param endtime  end time of the trial
*/

/*!The TRIAL structure is used to access a block of data within an EDF file 
  that is considered to be a trial within the experimental session. The start time 
  and end time of a TRIAL are defined using the edf_set_trial_identifier() function, 
  where a start and end message text token is specified.
*/
typedef struct {
	RECORDINGS *    rec;  /*!<recording information about the current trial*/
	unsigned int	duration;  /*!<duration of the current trial */
	unsigned int	starttime; /*!<start time of the trial */
	unsigned int	endtime;   /*!<end time of the trial */
} TRIAL;

/**@struct EDFFILE
	EDFFILE is a dummy structure that holds an EDF file handle.
	*/

typedef struct _EDFFILE EDFFILE;  /*!<EDFFILE is a dummy structure that holds an EDF file handle.*/


/* @struct BOOKMARK
    BOOKMARK is a dummy structure that holds a bookmark handle.
   	@param id
	
*/

/*!BOOKMARK is a dummy structure that holds a bookmark handle.
 */
typedef struct
{
	unsigned int id;
}BOOKMARK;

/*
#define FLOAT_TIME(x) (((double)((x)->time)) + (((x)->flags & SAMPLE_ADD_OFFSET)?0.5:0.0))
*/
/*=====================================================================*
 *																	   *
 *				GENERAL EDF DATA ACCESS FUNCTIONS					   *
 *																	   *
 *=====================================================================*/


/***********************************************************************
 *Function Name: edf_open_file
 *Input:		 file name, consistency, errval
 *Output: EDFFILE structure, pointer to int.
 *Purpose: opens the given edffile. If the return value is null then
 *		   the file could not be opened or the file is corrupt.
 *		   the errval is 0 if all the operation succeed. non zero
 *		   in other cases.
 ***********************************************************************/

/*!
   brief Opens the EDF file passed in by edf_file_name and preprocesses the EDF file. 
   @param fname name of the EDF file to be opened.
   @param consistency onsistency check control (for the time stamps of the start and end events, etc).
         	0, no consistency check.
    	1,  check consistency and report.
		2,  check consistency and fix.
   @param loadevents load/skip loading events 
		 0,  do not load events.
		1,  load events.
   @param loadsamples load/skip loading of samples
		 0,  do not load samples.
		1,  load samples.
   @param errval  This parameter is used for returning error value.
		The pointer should be a valid pointer to an integer. If the
		returned value is not  0 then an error occurred.

   @return if successful a pointer to EDFFILE structure is returned.
			Otherwise NULL is returned.
	*/

EDFFILE * EXPORT edf_open_file(const char *fname, int consistency,
							   int loadevents, int loadsamples,
							   int *errval);


 /***********************************************************************
 *Function Name: edf_close_file
 *Input:		 file name
 *Output: int
 *Purpose: closes file, frees all memory
 *		   returns an error code. if the error code is 0 then the
 *		   operation succeeded otherwise a non zero is returned.
 ***********************************************************************/
/*!  brief Closes an EDF file pointed to by the given EDFFILE pointer and releases all 
  of the resources (memory and physical file) related to this EDF file.
  @param ef a valid pointer to EDFFILE structure. This should be created by calling 
			edf_open_file ().
  @return  if successful it returns 0, otherwise a non zero is returned.*/



int EXPORT edf_close_file(EDFFILE * ef);










/***********************************************************************
 *Function Name: edf_get_next_data
 *Input:		 edffile
 *Output:		 int
 *Purpose:		 returns the data type of the next data. Possible
 *				 values are
 *				 STARTBLINK
 *				 STARTSACC
 *				 STARTFIX
 *				 STARTSAMPLES
 *				 STARTEVENTS
 *				 STARTPARSE
 *
 *				 ENDBLINK
 *				 ENDSACC
 *				 ENDFIX
 *				 ENDSAMPLES
 *				 ENDEVENTS
 *				 ENDPARSE
 *
 *				 FIXUPDATE
 *				 BREAKPARSE
 *				 BUTTONEVENT
 *				 INPUTEVENT
 *				 MESSAGEEVENT
 *
 *				 SAMPLE_TYPE
 *				 RECORDING_INFO
 *
 *				 NO_PENDING_ITEMS
 ***********************************************************************/
/*!
    brief Returns the type of the next data element in the EDF file pointed to by *edf. Each call to edf_get_next_data() 
	will retrieve the next data element within the data file. The contents of the data element are not accessed using 
	this method, only the type of the element is provided. Use edf_get_float_data() instead to access the contents of 
	the data element.
	@param ef a valid pointer to EDFFILE structure. This handle should be created by calling edf_open_file().
	@return One of the following values:
			 STARTBLINK   	 the upcoming data is a start blink event.n
			STARTSACC		 the upcoming data is a start saccade event.n
			 STARTFIX 		 the upcoming data is a start fixation event.n
			 STARTSAMPLES 	 the upcoming data is a start samples event.n
			 STARTEVENTS 	 the upcoming data is a start events event.n
			 STARTPARSE	     the upcoming data is a start parse event.n		
			 ENDBLINK		 the upcoming data is an end blink event.n
			 ENDSACC		     the upcoming data is an end saccade event.n
			 ENDFIX		     the upcoming data is an end fixation event.n
			 ENDSAMPLES	     the upcoming data is an end samples event.n
			 ENDEVENTS		 the upcoming data is an end events event.n
			 ENDPARSE		 the upcoming data is an end parse event.n
			 FIXUPDATE		 the upcoming data is a fixation update event.n
			BREAKPARSE	     the upcoming data is a break parse event.n
			 BUTTONEVENT	     the upcoming data is a button event.n
			 INPUTEVENT	     the upcoming data is an input event.n
		 MESSAGEEVENT	 the upcoming data is a message event.n
			 SAMPLE_TYPE 	 the upcoming data is a sample.n
			 RECORDING_INFO 	 the upcoming data is a recording info.n
			 NO_PENDING_ITEMS no more data left.*/
int EXPORT edf_get_next_data(EDFFILE *ef);


/***********************************************************************
 *Function Name: edf_get_float_data
 *Input:		 edffile
 *Output: int
 *Purpose: returns the float data with the type returned by
 *			edf_get_next_data
 ***********************************************************************/

/*!
   Returns the float data with the type returned by edf_get_next_data(). 
   This function does not move the current data access pointer to the next element; 
   use edf_get_next_data() instead to step through the data elements.
   @param ef a valid pointer to c EDFFILE structure. This handle should be created by 
			 calling edf_open_file().
   @return Returns a pointer to the c ALLF_DATA structure with the type returned by 
		   edf_get_next_data().*/

ALLF_DATA * EXPORT edf_get_float_data(EDFFILE *ef);
/**@}*/


/***********************************************************************
 *Function Name: edf_get_sample_close_to_time
 *Input:		 edffile
 *Output: int
 *Purpose: returns the float sample close to the time
 ***********************************************************************/
ALLF_DATA* EXPORT  edf_get_sample_close_to_time(EDFFILE *ef, unsigned int time);

/***********************************************************************
 *Function Name: edf_get_element_count
 *Input:		 edffile
 *Output: int
 *Purpose: returns the number of elements in the edf file.
 ***********************************************************************/
/*ingroup GENERALEDFDATAAccess
    Returns the number of elements (samples, eye events, messages, buttons, etc) in the EDF file.
	@param ef a valid pointer to c EDFFILE structure. This should be created by calling edf_open_file.
	@return  the number of elements in the c EDF file.*/
unsigned int EXPORT  edf_get_element_count(EDFFILE *ef);

/***********************************************************************
 *Function Name: edf_get_preamble_text
 *Input:		 edffile, pointer to char buffer, length of the buffer
 *Output: int
 *Purpose: copies the preamble text into the buffer.
 *		   if the preamble text is longer than the  length the
 *			text will be truncated.
 *			The returned content will always be null terminated
 ***********************************************************************/
/*ingroup GENERALEDFDATAAccess
    Copies the preamble text into the given buffer. If the preamble text is 
	longer than the length the text will be truncated. The returned content will always 
	be null terminated.
	@param ef a valid pointer to c EDFFILE structure. This handle should be created by 
			  calling edf_open_file().
    @param buffer a character array to be filled by the preamble text.
	@param length length of the buffer.
	@return returns c 0 if the operation is successful. */
int EXPORT  edf_get_preamble_text(EDFFILE *ef,
							char * buffer, int length);


/***********************************************************************
 *Function Name: edf_get_preamble_text_length
 *Input:		 edffile
 *Output: int
 *Purpose: Returns the preamble text length
 ***********************************************************************/
/*ingroup GENERALEDFDATAAccess
    Returns the length of the preamble text.
	@param edf  a valid pointer to c EDFFILE structure. This handle should be created by 
				calling edf_open_file().
	@return An integer for the length of preamble text.*/
int EXPORT  edf_get_preamble_text_length(EDFFILE * edf);






/***********************************************************************
 *Function Name: edf_get_revision
 *Input:		 edffile
 *Output: int
 *Purpose: returns the edffile revision
 ***********************************************************************/
int EXPORT edf_get_revision(EDFFILE *ef);


/***********************************************************************
 *Function Name: edf_get_eyelink_revision
 *Input:		 edffile
 *Output: int
 *Purpose: returns the revision of the tracker used to record the edf
 *		   file. ie. 1 for eyelinkI, 2 for eyelinkII, 3 for eyelinkCL
 ***********************************************************************/
int edf_get_eyelink_revision(EDFFILE *ef);



/*=====================================================================*
 *																	   *
 *						TRIAL RELATED FUNCTIONS						   *
 *																	   *
 *=====================================================================*/
/*defgroup TRIALRELATEDFunc Trial Related Functions
   The EDF access API also provides the following trial related functions for 
   the ease of counting the total number the trials in the recording file and 
   navigating between different trials.  To use this functionality, it is desirable 
   that the user first define the trial start/end identifier strings with 
   edf_set_trial_identifier(). [The identifier string settings can be checked with 
   the edf_get_start_trial_identifier() and edf_get_end_trial_identifier() functions].  
   Use edf_jump_to_trial(), edf_goto_previous_trial(), edf_goto_next_trial(), 
   edf_goto_trial_with_start_time(), or edf_goto_trial_with_end_time() functions 
   to go to a target trial.  The recording and start/end time of the target trial can be 
   checked with edf_get_trial_header().
   @{*/

/***********************************************************************
 *Function Name: edf_set_trial_identifier
 *Input:		 edffile, marker_string
 *Output: int
 *Purpose: sets the string that marks the beginning of the trial
 ***********************************************************************/
/*brief Sets the message strings that mark the beginning and the end of a trial. 
		  The message event that contains the marker string is considered start or 
		  end of the trial.
  @param edf a valid pointer to c EDFFILE structure. This should be created by 
			calling edf_open_file().
  @param start_marker_string string that contains the marker for beginning of a trial.
  @param end_marker_string string that contains the marker for end of the trial.
  @return c 0 if no error occurred.
  @remarks NOTE: The following restrictions apply for collecting the trials.n
				 1.The c start_marker_string message should be before the start recording 
				 (indicated by message  ``START'').n
				 2.The c end_marker_string message should be after the end recording 
				 (indicated by message  ``END'').n
				 3.If the c start_marker_string is not found before start recording or 
				 if the c start_marker_string is null, start recording will be the starting 
				 position of the trial.n
				 4.If the c end_marker_string is not found after the end recording, 
				 end recording will be the ending position of the trial.n
				 5.If c start_marker_string is not specified the string ``TRIALID'', 
				 if found, will be used as the c start_marker_string.n
				 6.If the c end_marker_string is not specified, the beginning of the 
				 next trial is the end of the current trial.*/					
int EXPORT  edf_set_trial_identifier(EDFFILE * edf,
									char *start_marker_string,
									char *end_marker_string);


/***********************************************************************
 *Function Name: edf_get_start_trial_identifier
 *Input:		 edffile
 *Output: int
 *Purpose: gets the string that marks the beginning of the trial
 ***********************************************************************/
/*brief Returns the trial identifier that marks the beginning of a trial.
	@param ef a valid pointer to c EDFFILE structure. This should be created by 
	calling edf_open_file().
	@return a string that marks the beginning of a trial.*/
char* EXPORT edf_get_start_trial_identifier(EDFFILE * ef);

/***********************************************************************
 *Function Name: edf_get_end_trial_identifier
 *Input:		 edffile
 *Output: int
 *Purpose: gets the string that marks the beginning of the trial
 ***********************************************************************/
/*brief Returns the trial identifier that marks the end of a trial.
 @param ef a valid pointer to c EDFFILE structure. This should be created by calling 
		   edf_open_file().
 @return a string that marks the end of a trial.*/
char* EXPORT  edf_get_end_trial_identifier(EDFFILE * ef);

/***********************************************************************
 *Function Name: edf_get_trial_count
 *Input:		 edffile
 *Output: int
 *Purpose: returns the number of trials
 ***********************************************************************/
/*!brief Returns the number of trials in the EDF file.
	@param edf a valid pointer to c EDFFILE structure. This should be created 
		   by calling edf_open_file().
   @return an integer for the number of trials in the EDF file.*/
int EXPORT  edf_get_trial_count(EDFFILE *edf);


/***********************************************************************
 *Function Name: edf_jump_to_trial
 *Input:		 edffile
 *Output: int
 *Purpose: jumps to the beginning of a given trial.
 ***********************************************************************/
/*! Jumps to the beginning of a given trial. 
	@param edf a valid pointer to c EDFFILE structure. This should be created 
			by calling edf_open_file().
	@param trial trial number.  This should be a value between c 0 and 
			edf_get_trial_count ()- c 1.
	@return unless there are any errors it returns a 0.*/
int EXPORT  edf_jump_to_trial(EDFFILE * edf, int trial);




/***********************************************************************
 *Function Name: edf_get_trial_headers
 *Input:		 edffile
 *Output: int
 *Purpose: Returns the current trial information
 ***********************************************************************/
/*!brief Returns the trial specific information. See the TRIAL structure for 
	more details.
	@param edf a valid pointer to c EDFFILE structure. This should be created 
				by calling edf_open_file().
	@param trial pointer to a valid c TRIAL structure (note c trial must be 
				initialized before being used as a parameter for this function).  
				This pointer is used to hold information of the current trial.

	@return unless there are any errors it returns a 0.*/

int EXPORT edf_get_trial_header(EDFFILE * edf,TRIAL *trial);




/***********************************************************************
 *Function Name: edf_goto_previous_trial
 *Input:		 edffile
 *Output: int
 *Purpose: moves to the previous trial
 ***********************************************************************/
/*!brief Jumps to the beginning of the previous trial.
	@param edf a valid pointer to c EDFFILE structure. This should be created 
				by calling edf_open_file().
	@return unless there are any errors it returns c 0.*/
int EXPORT  edf_goto_previous_trial(EDFFILE * edf);


/***********************************************************************
 *Function Name: edf_goto_next_trial
 *Input:		 edffile
 *Output: int
 *Purpose: moves to the next trial
 ***********************************************************************/
/*!brief Jumps to the beginning of the next trial.
	@param edf a valid pointer to c EDFFILE structure. 
				This should be created by calling edf_open_file().
	@return unless there are any errors it returns c 0.*/
int EXPORT edf_goto_next_trial(EDFFILE * edf);


/***********************************************************************
 *Function Name: edf_goto_trial_with_start_time
 *Input:		 edffile, start_time
 *Output: int
 *Purpose: moves to the trial with the given start time
 ***********************************************************************/
/*!brief Jumps to the trial that has the same start time as the given start time.
	@param edf a valid pointer to c EDFFILE structure. This should be created by 
				calling edf_open_file().
	@return unless there are any errors it returns 0.*/
int EXPORT edf_goto_trial_with_start_time(EDFFILE * edf,
										  unsigned int start_time);

/***********************************************************************
 *Function Name: edf_goto_trial_with_end_time
 *Input:		 edffile, end_time
 *Output: int
 *Purpose: moves to the trial with the given end time
 ***********************************************************************/
/*! brief Jumps to the trial that has the same start time as the given end time.
	@param edf a valid pointer to c EDFFILE structure. This should be created by 
				calling edf_open_file().
	@return unless there are any errors it returns c 0. */
int EXPORT edf_goto_trial_with_end_time(EDFFILE * edf,
										  unsigned int end_time);
/**@}*/







/*=====================================================================*
 *																	   *
 *					BOOKMARK RELATED FUNCTIONS						   *
 *																	   *
 *=====================================================================*/

/*!defgroup BOOKMARK Bookmark Related Functions
In addition to navigation between different trials in an EDF recording file 
with the functions provided in the previous section, the EDF access API also 
allows the user to ``bookmark'' any position of the EDF file using the edf_set_bookmark() 
function.  The bookmarks can be revisited with edf_goto_bookmark().  Finally, the bookmarks 
should be freed with the edf_free_bookmark() function call.
@{*/

/***********************************************************************
 *Function Name: edf_set_bookmark
 *Input:		 edffile, pointer to bookmark object
 *Output: int
 *Purpose: mark the current position of edffile
 ***********************************************************************/
/*! Bookmark the current position of the edf file.
	@param ef a valid pointer to c EDFFILE structure. This should be created 
				by calling edf_open_file.
	@param bm pointer to a valid c BOOKMARK structure. This structure will be 
				filled by this function.  c bm should be initialized before 
				being used by this function.
	@return unless there are any errors it returns 0.*/
int EXPORT  edf_set_bookmark(EDFFILE *ef, BOOKMARK *bm);

/***********************************************************************
 *Function Name: edf_free_bookmark
 *Input:		 edffile, pointer to bookmark object
 *Output: int
 *Purpose: remove the bookmark
 ***********************************************************************/
/*! Removes an existing bookmark
	@param ef a valid pointer to c EDFFILE structure. This should be created 
				by calling edf_open_file.
	@param bm pointer to a valid c BOOKMARK structure. This structure will be 
				filled by this function.  Before calling this function edf_set_bookmark 
				should be called and bm should be initialized there.
	@return unless there are any errors it returns 0.*/
int EXPORT edf_free_bookmark(EDFFILE *ef, BOOKMARK *bm);


/***********************************************************************
 *Function Name: edf_goto_bookmark
 *Input:		 edffile, pointer to bookmark object
 *Output: int
 *Purpose: jump to the bookmark
 ***********************************************************************/
/*! Jumps to the given bookmark.
	@param ef a valid pointer to c EDFFILE structure. This should be created by calling 
				edf_open_file.
	@param bm pointer to a valid c BOOKMARK structure. This structure will be filled 
				by this function.  Before calling this function edf_set_bookmark should 
				be called and bm should be initialized there.
	@return unless there are any errors it returns c 0.*/
int EXPORT edf_goto_bookmark(EDFFILE *ef, BOOKMARK *bm);

/**@}*/
/***********************************************************************
 *Function Name: edf_goto_next_bookmark
 *Input:		 edffile
 *Output: int
 *Purpose: jump to the next bookmark
 ***********************************************************************/
int EXPORT edf_goto_next_bookmark(EDFFILE *ef);

/***********************************************************************
 *Function Name: edf_goto_previous_bookmark
 *Input:		 edffile
 *Output: int
 *Purpose: jump to the previous bookmark
 ***********************************************************************/
int EXPORT edf_goto_previous_bookmark(EDFFILE *ef);



/***********************************************************************
 *Function Name: edf_get_version
 *Input:		 none
 *Output: char *
 *Purpose: returns the version of edfapi
 ***********************************************************************/
/*!defgroup EDFSpecificFunc EDF Specific Functions
 * @{
 */
/*!Returns a string which indicates the version of EDFAPI.dll library used.
 * @return	a string indicating the version of EDFAPI library used.*/
char * EXPORT edf_get_version();

/**@}*/ 


/***********************************************************************
 *Function Name: get_event
 *Input:		 ALLF_DATA *
 *Output: FEVENT *
 *Purpose: convenient function to translate an ALLF_DATA union to FEVENT
 ***********************************************************************/
FEVENT     *  EXPORT edf_get_event(ALLF_DATA *allfdata);

/***********************************************************************
 *Function Name: get_event
 *Input:		 ALLF_DATA *
 *Output: FEVENT *
 *Purpose: convenient function to translate an ALLF_DATA union to FSAMPLE
 ***********************************************************************/
FSAMPLE    *  EXPORT edf_get_sample(ALLF_DATA *allfdata);

/***********************************************************************
 *Function Name: get_event
 *Input:		 ALLF_DATA *
 *Output: FEVENT *
 *Purpose: convenient function to translate an ALLF_DATA union to RECORDINGS
 ***********************************************************************/
RECORDINGS *  EXPORT edf_get_recording(ALLF_DATA *allfdata);




/*








		*/
/*
ELCL
*/
void EXPORT edf_get_uncorrected_raw_pupil(EDFFILE *edf,FSAMPLE *sam, int eye,float *rv);
void EXPORT edf_get_uncorrected_raw_cr(EDFFILE *edf,FSAMPLE *sam, int eye,float *rv);
UINT32   EXPORT edf_get_uncorrected_pupil_area(EDFFILE *edf,FSAMPLE *sam, int eye);
UINT32   EXPORT edf_get_uncorrected_cr_area(EDFFILE *edf,FSAMPLE *sam, int eye);
void EXPORT edf_get_pupil_dimension(EDFFILE *edf,FSAMPLE *sam, int eye, UINT32 *rv);
void EXPORT edf_get_cr_dimension(EDFFILE *edf,FSAMPLE *sam, UINT32 *rv);
void EXPORT edf_get_window_position(EDFFILE *edf,FSAMPLE *sam, UINT32 *rv);
void EXPORT edf_get_pupil_cr(EDFFILE *edf,FSAMPLE *sam, int eye, float *rv);
UINT32   EXPORT edf_get_uncorrected_cr2_area(EDFFILE *edf,FSAMPLE *sam, int eye);
void EXPORT edf_get_uncorrected_raw_cr2(EDFFILE *edf,FSAMPLE *sam, int eye,float *rv);




/*
equivalent functions to edf_get_float_data
*/
FEVENT * edf_get_event_data(EDFFILE *edf);
FSAMPLE * edf_get_sample_data(EDFFILE *edf);
RECORDINGS * edf_get_recording_data(EDFFILE *edf);


/*
Redirect log support
*/
void edf_set_log_function(void (*lfcn )(char *log));

#ifdef __cplusplus
};
#endif

#ifdef __cplusplus
/*=====================================================================*
 *																	   *
 *							C++ EDF WRAPPER 						   *
 *																	   *
 *=====================================================================*/


class EDF
{
protected:
	EDFFILE * edf;
public:
/*=====================================================================*
 *																	   *
 *				GENERAL EDF DATA ACCESS FUNCTIONS					   *
 *																	   *
 *=====================================================================*/
	inline EDF(const char *fname, int consistency, int loadevents,
		int loadsamples, int *errval);
	virtual inline ~EDF();
	inline int getEyelinkCRMode();
	inline int getNextData();
	inline ALLF_DATA * getFloatData();
	inline int getPreambleText(char * buffer, int length);
	inline int getPreambleTextLength();
	inline int getSampleInterval();
	inline int getRevision();

	inline int getElementCount() { return edf_get_element_count(edf); }

/*=====================================================================*
 *																	   *
 *						TRIAL RELATED FUNCTIONS						   *
 *																	   *
 *=====================================================================*/
	inline int setTrialIdentifier(char *start_string, char *end_string);
	inline int getTrialCount();
	inline int jumpToTrial(int trial);
	inline int getTrialHeader(TRIAL *trial);
	inline int gotoPreviousTrial();
	inline int gotoNextTrial();
	inline int gotoTrialWithStartTime(unsigned int start_time);
	inline int gotoTrialWithEndTime(unsigned int end_time);

/*=====================================================================*
 *																	   *
 *					BOOKMARK RELATED FUNCTIONS						   *
 *																	   *
 *=====================================================================*/
	inline int setBookmark(BOOKMARK *bm);
	inline int freeBookmark(BOOKMARK *bm);
	inline int gotoBookmark(BOOKMARK *bm);
	inline int gotoNextBookmark();
	inline int gotoPreviousBookmark();
    static inline FEVENT * getEvent(ALLF_DATA *allfdata);
    static inline FSAMPLE   * getSample(ALLF_DATA *allfdata);
    static inline RECORDINGS * getRecording(ALLF_DATA *allfdata);
    inline FEVENT * getEvent();
    inline FSAMPLE   * getSample();
    inline RECORDINGS * getRecording();

};



inline EDF::EDF(const char *fname, int consistency, int loadevents,
				int loadsamples, int *errval)
{
	edf = edf_open_file(fname,consistency, loadevents, loadsamples,
		errval);
}
inline EDF::~EDF()
{
	if(edf)
		edf_close_file(edf);

}


inline int EDF::getNextData()
{
	return edf_get_next_data(edf);
}
inline ALLF_DATA * EDF::getFloatData()
{
	return edf_get_float_data(edf);
}
inline int EDF::getPreambleText(char * buffer, int length)
{
	return edf_get_preamble_text(edf,buffer,length);
}
inline int EDF::getPreambleTextLength()
{
	return edf_get_preamble_text_length(edf);
}

inline int EDF::getRevision()
{
	return edf_get_revision(edf);
}

inline int EDF::setTrialIdentifier(char *start_marker_string, char * end_marker_string)
{
	return edf_set_trial_identifier(edf,start_marker_string, end_marker_string);
}


inline int EDF::getTrialCount()
{
	return edf_get_trial_count(edf);
}
inline int EDF::jumpToTrial(int trial)
{
	return edf_jump_to_trial(edf,trial);
}
inline int EDF::getTrialHeader(TRIAL *trial)
{
	return edf_get_trial_header(edf,trial);
}
inline int EDF::gotoPreviousTrial()
{
	return edf_goto_previous_trial(edf);
}
inline int EDF::gotoNextTrial()
{
	return edf_goto_next_trial(edf);
}
inline int EDF::gotoTrialWithStartTime(unsigned int start_time)
{
	return edf_goto_trial_with_start_time(edf, start_time);
}
inline int EDF::gotoTrialWithEndTime(unsigned int end_time)
{
	return edf_goto_trial_with_end_time(edf, end_time);
}

inline int EDF::setBookmark(BOOKMARK *bm)
{
	return edf_set_bookmark(edf, bm);
}
inline int EDF::freeBookmark(BOOKMARK *bm)
{
	return edf_free_bookmark(edf,bm);
}
inline int EDF::gotoBookmark(BOOKMARK *bm)
{
	return edf_goto_bookmark(edf,bm);
}
inline int EDF::gotoNextBookmark()
{
	return edf_goto_next_bookmark(edf);
}
inline int EDF::gotoPreviousBookmark()
{
	return edf_goto_previous_bookmark(edf);
}




inline FEVENT * EDF::getEvent(ALLF_DATA *allfdata)
{
    if(allfdata->fe.type == SAMPLE_TYPE || allfdata->fe.type == RECORDING_INFO)
        return 0L;
    else
        return (FEVENT *)allfdata;
}

inline FSAMPLE   * EDF::getSample(ALLF_DATA *allfdata)
{
    if(allfdata->fe.type == SAMPLE_TYPE)
        return (FSAMPLE *)allfdata;
    else
        return 0L;

}
inline RECORDINGS * EDF::getRecording(ALLF_DATA *allfdata)
{
    if(allfdata->fe.type == RECORDING_INFO)
        return (RECORDINGS *)allfdata;
    else
        return 0L;

}

inline FEVENT * EDF::getEvent()
{
    ALLF_DATA *allfdata = this->getFloatData();
    return EDF::getEvent(allfdata);
}

inline FSAMPLE   * EDF::getSample()
{
    ALLF_DATA *allfdata = this->getFloatData();
    return EDF::getSample(allfdata);
}
inline RECORDINGS * EDF::getRecording()
{
    ALLF_DATA *allfdata = this->getFloatData();
    return EDF::getRecording(allfdata);
}

#endif







import os
import logging 

# For Converting Dates
import matplotlib.dates as mpldate

def GetSerialDate( normal_date ):
    return mpldate.date2num(normal_date)

def GetSerialDateFromString( str_name ):
    return mpldate.datestr2num(str_name) 

def MakeDir(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)

def GetLogger(logName):
    myLog = logging.getLogger(logName)
    if not myLog.hasHandlers():
        myLog.setLevel(logging.DEBUG)
        h = logging.FileHandler("..\\LogFile\\logFile_storm_"+logName, 'a')
        f = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s %(message)s')
        h.setFormatter(f)
        myLog.addHandler(h)
    return myLog
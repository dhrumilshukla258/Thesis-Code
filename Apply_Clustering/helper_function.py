import logging 

def GetLogger(logName):
    myLog = logging.getLogger(logName)
    if not myLog.hasHandlers():
        myLog.setLevel(logging.DEBUG)
        h = logging.FileHandler("..\\LogFile\\logFile_cluster_"+logName, 'a')
        f = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s %(message)s')
        h.setFormatter(f)
        myLog.addHandler(h)
    return myLog
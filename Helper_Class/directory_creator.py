import os
import json
import logging
import time

logging.basicConfig(filename='..\\LogFile\\logFile_directory_create',level=logging.DEBUG)

path = "..\\..\\MyCreatedData"
data = json.load( open("..\\files_per_year_without_bestTrack.json") )

def MakeDirectory(f_path):
    try:
        os.mkdir(f_path)
    except OSError:
        logging.debug ("Creation of the directory %s failed" % f_path)
    else:
        logging.debug ("Successfully created the directory %s " % f_path)


startMain = time.time()
logging.debug(" StartTime:- " + str(startMain) )
for k1,v1 in data.items():
    year_path = path + "\\" + k1
    
    MakeDirectory(year_path)

    for k2,v2 in v1.items():
        region_path = year_path +"\\" + k2
        MakeDirectory(region_path)
        
        for k3,v3 in v2.items():
            storm_path = region_path + "\\" + k3
            MakeDirectory(storm_path)
            
            for k4,v4 in v3.items():
                if "BestTrack" != k4:
                    f_path = storm_path + "\\" + k4
                    MakeDirectory(f_path)

                    #Swath 1
                    swath_path = f_path + "\\S1"
                    MakeDirectory(swath_path)
                    freq19H = swath_path + "\\19H"
                    MakeDirectory(freq19H)
                    freq19V = swath_path + "\\19V"
                    MakeDirectory(freq19V)
                    freq22V = swath_path + "\\22V"
                    MakeDirectory(freq22V)
                    
                    #Swath 2
                    swath_path = f_path + "\\S2"
                    MakeDirectory(swath_path)
                    freq37H = swath_path + "\\37H"
                    MakeDirectory(freq37H)
                    freq37V = swath_path + "\\37V"
                    MakeDirectory(freq37V)
                    
                    #Swath 3
                    swath_path = f_path + "\\S3"
                    MakeDirectory(swath_path)
                    freq150H = swath_path + "\\150H"
                    MakeDirectory(freq150H)
                    freq183_1H = swath_path + "\\183_1H"
                    MakeDirectory(freq183_1H)
                    freq183_3H = swath_path + "\\183_3H"
                    MakeDirectory(freq183_3H)
                    freq183_7H = swath_path + "\\183_7H"
                    MakeDirectory(freq183_7H)
                    
                    #Swath 4
                    swath_path = f_path + "\\S4"
                    MakeDirectory(swath_path)
                    freq91H = swath_path + "\\91H"
                    MakeDirectory(freq91H)
                    freq91V = swath_path + "\\91V"
                    MakeDirectory(freq91V)
endMain = time.time()
logging.debug(" EndTime:- " + str(endMain) )                   
logging.debug( " Total time:- " + str( (endMain - startMain)/3600 ) + " hours" )
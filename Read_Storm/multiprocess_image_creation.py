# Reading json file
import json

# For Parallizing the Creation of Images
from multiprocessing import Pool, cpu_count

# Get current process details
from multiprocessing import current_process

import time
from storm_reader import StormReader
from helper_function import GetLogger


def RunThreads(stormDict, region,year,stormNo):
    sR = StormReader(stormDict, region,year,stormNo)
    #print(region,year,stormNo)
    sR.ReadStorm()

if __name__ == '__main__': 
    data = json.load( open( "..\\valid_matfiles_and_best_track.json" ) )

    arguments = []
    for region,v1 in data.items():
        for year,v2 in v1.items():
            for stormNo,stormDict in v2.items():
                arguments.append([stormDict,region,year,stormNo])
    start = time.time()
    
    # Processes according to total cores available
    pool = Pool(processes=cpu_count())
    pool.starmap(RunThreads, arguments)
    pool.close()
    
    myLog = GetLogger(current_process().name)
    myLog.debug("Total Time taken: "+str( (time.time()-start)/3600 ) + " hours")
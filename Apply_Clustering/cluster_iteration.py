# from multiprocessing import Pool, cpu_count

# Get current process details
from multiprocessing import current_process

import time
import pandas as pd
from test_cases import TestCases
from helper_function import GetLogger

'''
def RunThreads(df,region,freq):    
    test.RunTests(df,region,freq)
'''

if __name__ == '__main__':
    #,'CPAC','EPAC','IO','SHEM','WPAC'
    imgFreq = {}
    fList = ['19H','19V','19','22V','37V','37H','37','91H','91V','91','150H','183_1H','183_3H','183_7H']
    rList = ['ATL']
    for r in rList:
        imgFreq[r] = {}
        for f in fList:
            imgFreq[r][f] = pd.read_csv("..//ImagesPerFreq//"+r+"_"+f+'.csv')
    
    '''
    arguments = []
    for region,v1 in imgFreq.items():
        for freq, _df in v1.items():
            arguments.append([_df,region,freq])
    '''
    
    start = time.time()
    
    test = TestCases()
    for region,v1 in imgFreq.items():
        for freq, _df in v1.items():
            test.RunTests(_df,region,freq)
    
    '''
    # Processes according to total cores available
    pool = Pool(processes=cpu_count()-1)
    pool.starmap(RunThreads, arguments)
    pool.close()
    '''
    
    myLog = GetLogger(current_process().name)
    myLog.debug("Total Time taken: "+str( (time.time()-start)/3600 ) + " hours")
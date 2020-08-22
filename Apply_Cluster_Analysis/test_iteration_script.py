import os
import time
import shutil
import logging
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from significant_cluster import SignificantCluster
from test_result_script import TestResults
from test_result_script import MakeDir

# For Parallizing the Reading of Images
from multiprocessing import Pool, cpu_count
from multiprocessing import current_process

#
#
#'kmeans','fuzzykmeans'
funList = ['agglo']

mapping_values = {}
mapping_values[0.0] = 0.0
mapping_values[1.0] = 1.0
mapping_values[2.0] = 2.0
mapping_values[3.0] = 3.0
mapping_values[4.0] = 4.0
mapping_values[5.0] = 5.0
mapping_values[6.0] = 6.0
mapping_values[7.0] = 7.0

mapping_values[0.5] = 0.0
mapping_values[1.5] = 1.0
mapping_values[2.5] = 2.0
mapping_values[3.5] = 3.0
mapping_values[4.5] = 4.0
mapping_values[5.5] = 5.0
mapping_values[6.5] = 6.0
mapping_values[7.5] = 7.0

def MakeDir(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)

def GetLogger(logName):
    myLog = logging.getLogger(logName)
    if not myLog.hasHandlers():
        myLog.setLevel(logging.DEBUG)
        h = logging.FileHandler("logFile_cluster_"+logName, 'a')
        f = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s %(message)s')
        h.setFormatter(f)
        myLog.addHandler(h)
    return myLog

def Create_Grouped_TNo_Column(cl_path):
    test_df = pd.read_csv( cl_path+"testInfo.csv" )
    test_df["T_No_NonGrouped"] = test_df.T_No
    test_df[ "T_No" ] = test_df["T_No_NonGrouped"].map(mapping_values)
    test_df.to_csv(cl_path+"testInfo.csv",index=False)

def RunTestAnalysis( args ):
    mLog = GetLogger(current_process().name)
    for reg,freq in args:
        path = "E:\\AllFrequencies\\"+reg+"\\"+freq+"\\"
        for fun in funList:
            funpath = path + fun +"\\"

            test_list = os.listdir(funpath)
            for test in test_list:
                if ".png" in test or "NonGroupedResult" in test:
                    continue

                total_sil_mean = []
                total_sig_clusters = []
                total_images = []
                total_k = []
                
                testpath = funpath + test + "\\"
                cl_nums = os.listdir( funpath+test )
                for cl_num in cl_nums:

                    cl_path = testpath+cl_num+"\\"
                    cl_df = pd.read_csv(cl_path+"sig_testInfo.csv")

                    #tR = TestResults(cl_df, cl_path, freq)
                    #tR.Composite_Images()
                    #tR.Create_Histogram()
                    #tR.Sil_Distribution( freq, cl_num )
                    #tR.Remove_Pasted_Images()
                    #del tR

                    total_images.append(len(cl_df))
                    total_sig_clusters.append( int(len(os.listdir(cl_path+"Composite_Images\\"))/2) )
                    total_sil_mean.append(np.mean( cl_df.SilhouetteVal ))
                    total_k.append(int(cl_num))

                fig, ax = plt.subplots( ncols=1, nrows=3, figsize=(8,10), sharex=True )
                ax[0].set_title(test)
                ax[0].set_ylabel("Total Images")
                ax[0].set_xlabel("Cluster Number")
                ax[0].scatter(total_k,total_images)

                ax[1].set_ylabel("Total Sig Cluster")
                ax[1].set_xlabel("Cluster Number")
                ax[1].scatter(total_k,total_sig_clusters)

                ax[2].set_ylabel("Mean Sil Value")
                ax[2].set_xlabel("Cluster Number")
                ax[2].scatter(total_k,total_sil_mean)

                plt.savefig( funpath+test+"_cluster_compare.png")
                plt.close()

#continue

'''

mLog.debug(testpath)

cl_nums = os.listdir( funpath+test )
for cl_num in cl_nums:
    cl_path = testpath+cl_num+"\\"
    if os.path.isfile( cl_path+"sig_testInfo.csv" ) == False:
        #Create_Grouped_TNo_Column(cl_path)
        sC = SignificantCluster(pd.read_csv( cl_path+"testInfo.csv" ), cl_path)
        sC.Create_Sig_Cluster_DF()
        del sC
    
    cl_df = pd.read_csv(cl_path+"sig_testInfo.csv")

    
'''
def DivideDataForMultipleProcess():
    fList = ['19H','19V','19PCT','22V','37V','37H','37PCT','91H','91V','91PCT','150H','183_1H','183_3H','183_7H']
    rList = ['ATL','CPAC','EPAC','IO','SHEM','WPAC']
    # Here we could have used multiprocess manager but thats slow
    # so here is another trick which is to distribute data equally to all processes
    reg_freq_dict = {}
    
    #Dividing data based on total number of processes == if total cores are 4 
    #we get 21 (reg,freq) combo allocated to each processes
    total_allocation_of_argument_on_each_process = len(rList)*len(fList) / cpu_count()
    print(total_allocation_of_argument_on_each_process)
    i = 0
    j = 0
    for r in rList:
        for f in fList:
            if reg_freq_dict.get(j) == None:
                reg_freq_dict[j] = []
            reg_freq_dict[ j ].append( (r,f) )
            
            i+=1
            if i >= total_allocation_of_argument_on_each_process:
                i = 0
                j+=1
    
    # variable j should be equal to cpu_count()
    arguments = []
    for i,arg in reg_freq_dict.items():
        arguments.append(arg)
    
    return arguments

if __name__ == '__main__':
 
    # arguments length will be equal to cpu_count()
    arguments = DivideDataForMultipleProcess()
    
    print("===================================================")
    s = time.time()
     
    # Processes according to total cores available
    print(cpu_count(),len(arguments))
    pool = Pool(processes=cpu_count())
    pool.map(RunTestAnalysis, arguments)
    pool.close()
    
    print("Total time taken : "+str( (time.time()-s)/60 )+" min")
    print("=================================================")



'''
if ".png" in test:
os.rename(funpath+test,funpath+test[:6]+"_cluster_compare.png")
'''

                
                
            

'''
ls = os.listdir(funpath)
            for test in ls:
                testPath = funpath + test + "\\"
                if os.path.isfile( testPath+"sig_testInfo.csv" ) == False:
                    sC = SignificantCluster(pd.read_csv( testPath+"testInfo.csv" ), testPath)
                    sC.Create_Sig_Cluster_DF()
                    del sC
                imgFreq[reg][freq][fun][test] = pd.read_csv(testPath+"sig_testInfo.csv")
'''



'''
        total_sil_mean = []
        total_sig_clusters = []
        total_images = []
        total_k = []

        cl_nums = os.listdir( funpath+test )
        for cl_num in cl_nums:
            cl_path = testpath+cl_num+"\\"

            if os.path.isfile( cl_path+"sig_testInfo.csv" ) == False:
                sC = SignificantCluster(pd.read_csv( cl_path+"testInfo.csv" ), cl_path)
                sC.Create_Sig_Cluster_DF()
                del sC

            #imgFreq[reg][freq][fun][test] = pd.read_csv(cl_path+"sig_testInfo.csv")
            cl_df = pd.read_csv(cl_path+"sig_testInfo.csv")

            tR = TestResults(cl_df, cl_path, freq)
            tR.Composite_Images()
            tR.Create_Histogram()
            tR.Sil_Distribution( freq, cl_num )
            #tR.Remove_Pasted_Images()
            del tR

            total_images.append(len(cl_df))
            total_sig_clusters.append( int(len(os.listdir(cl_path+"Composite_Images\\"))/2) )
            total_sil_mean.append(np.mean( cl_df.SilhouetteVal ))
            total_k.append(int(cl_num))

        fig, ax = plt.subplots( ncols=1, nrows=3, figsize=(8,10), sharex=True )
        ax[0].set_title(test)
        ax[0].set_ylabel("Total Images")
        ax[0].set_xlabel("Cluster Number")
        ax[0].scatter(total_k,total_images)

        ax[1].set_ylabel("Total Sig Cluster")
        ax[1].set_xlabel("Cluster Number")
        ax[1].scatter(total_k,total_sig_clusters)

        ax[2].set_ylabel("Mean Sil Value")
        ax[2].set_xlabel("Cluster Number")
        ax[2].scatter(total_k,total_sil_mean)

        plt.savefig( funpath+test+"_cluster_compare.png")
        plt.close()
        '''


'''
non_grouped_path =  cl_path+"NonGroupedResult\\"
MakeDir(non_grouped_path)

# All the Files Created for each test Cases
test_file_and_dir_list = os.listdir(cl_path)
for folder_or_file_name in test_file_and_dir_list:
    if "dend" in folder_or_file_name or "testInfo.csv" == folder_or_file_name or "NonGroupedResult" == folder_or_file_name:
        continue

    old_path = cl_path+folder_or_file_name
    new_path = non_grouped_path+folder_or_file_name
    shutil.move(old_path,new_path)
'''
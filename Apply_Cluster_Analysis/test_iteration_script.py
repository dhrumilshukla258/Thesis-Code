import os
import re
import cv2
import time
import json
import shutil
import logging
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from significant_cluster import SignificantCluster
from test_result_script import TestResults
from test_result_script import MakeDir

# For Parallizing the Reading of (Region and Freq)
from multiprocessing import Pool, cpu_count
from multiprocessing import current_process

#
#
#'kmeans','fuzzykmeans','agglo'
funList = [ ['agglo',['NonGroupedResult\\','GroupedResult\\']],#
            ['kmeans',['']],
            ['fuzzykmeans',['']]
          ]

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

def CreateGraphicalComparisionBetweenClusterSize(funpath,g_ng_for_agglo_blank_for_other_algo):
    test_list = os.listdir( funpath )
    
    for sig in g_ng_for_agglo_blank_for_other_algo:
        for test in test_list:
            if ".png" in test or "NonGroupedResult" in test or "GroupedResult" in test:            
                continue
            
            # Get necessary data
            total_sil_mean = []
            total_sig_clusters = []
            total_images = []
            total_k = []
            
            # Setting test path
            testpath = funpath + test + "\\"
            
            # Getting all cluster sizes from the test path
            cl_nums = os.listdir( testpath )
            cl_nums.sort()
            
            # Iterate over all cluster sizes
            for cl_num in cl_nums:
                cl_path = testpath+cl_num+"\\"
                sig_path = cl_path+sig
                sig_cl_df = pd.read_csv(sig_path+"sig_testInfo.csv")

                total_images.append(len(sig_cl_df))
                total_sig_clusters.append( int(len(os.listdir(sig_path+"Composite_Images\\"))/2) )
                total_sil_mean.append(np.mean( sig_cl_df.SilhouetteVal ))
                total_k.append(int(cl_num))

            # Plotting the dataset
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

            plt.savefig( funpath+sig+"\\"+test+"_cluster_compare.png")
            plt.close()

def CreateTableOfClusterComposite(funpath,g_ng_for_agglo_blank_for_other_algo):
    def GetMaxComposites(testpath, whichResult):
        cl_sizes = os.listdir( testpath )
        cl_sizes.sort()
        total_sig_clusters = []
        for cl_size in cl_sizes:
            cl_path = testpath+cl_size+"\\"
            # nongrouped | grouped
            sig_path = cl_path+whichResult+"\\Composite_Images\\"
            compo_imgs = os.listdir( sig_path )
            total_sig_clusters.append( int(len(compo_imgs)/2) )
        return max(total_sig_clusters)

    #mLog = GetLogger(current_process().name)
    test_list = os.listdir( funpath )
    for sig in g_ng_for_agglo_blank_for_other_algo#["NonGroupedResult", "GroupedResult",""]:
        for test in test_list:
            if ".png" in test or "NonGroupedResult" in test or "GroupedResult" in test:            
                continue
            
            # Setting test path
            testpath = funpath + test + "\\"
            
            # Get total maximum sig clusters
            max_total_sig_clusters = GetMaxComposites(testpath,sig)

            # Getting all cluster sizes from the test path
            cl_nums = os.listdir( testpath )
            cl_nums.sort()
            
            # Setting the Table to merge all images in the list
            arr = [[]] * len(cl_nums) # This will be from 10-50

            # Iterate over all cluster sizes
            for cl_num in cl_nums:
                cl_path = testpath+cl_num+"\\"
                sig_path = cl_path+sig
                sig_cl_df = pd.read_csv(sig_path+"sig_testInfo.csv")

                # The size of individual images is fixed | This will store all the significant composites
                # for a unique cl_num in the list
                sig_cluster_arr = [np.ones((96,96,3), dtype=np.uint8)] * (max_total_sig_clusters+1)
                img = cv2.putText(np.ones((96,96,3), dtype=np.uint8), text=str(cl_num), org=(38,38),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255),
                    thickness=2, lineType=cv2.LINE_AA)
                sig_cluster_arr[0] = img
                j=1
                compo_imgs = os.listdir( sig_path+"Composite_Images\\" )
                for img_name in compo_imgs:
                    if "Max" in img_name:
                        continue
                    
                    # The image name consist of sig_cl_number(where image belongs) and total images in that sig_cl_number
                    index = img_name.find("_")
                    sig_cl_num = img_name[:index]
                    avg_sil_score = np.mean( sig_cl_df[ sig_cl_df.ClusterLabel == int(sig_cl_num) ].SilhouetteVal )
                    
                    total_images = img_name[index+1:-4]
                    img = cv2.imread(sig_path+"Composite_Images\\"+img_name)
                    try:
                        img = img[:,:480]
                    except TypeError:
                        continue

                    img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
                    # Labeling the image                    
                    img = cv2.putText(img, text=total_images, org=(10,20),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255),
                    thickness=2, lineType=cv2.LINE_AA)

                    img = cv2.putText(img, text="{:.2f}".format( avg_sil_score ), org=(10,90),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255),
                    thickness=2, lineType=cv2.LINE_AA)

                    sig_cluster_arr[j] = img
                    j+=1
                
                arr[int(cl_num)-10] = cv2.hconcat(sig_cluster_arr)
        
            #funpath+ | sig+"\\"+ | test
            cv2.imwrite(funpath+test+"_CompositeGraph.png", cv2.vconcat(arr) )

def AnalyzeEachClusterSizes(funpath, freq, g_ng_for_agglo_blank_for_other_algo):
    test_list = os.listdir( funpath )

    for sig in g_ng_for_agglo_blank_for_other_algo: #["NonGroupedResult", "GroupedResult",""]:
        for test in test_list:
            if ".png" in test or "NonGroupedResult" in test or "GroupedResult" in test:            
                continue

            # Setting test path
            testpath = funpath + test + "\\"

            # Getting all cluster sizes from the test path
            cl_nums = os.listdir( testpath )
            cl_nums.sort()

            # Iterate over all cluster sizes
            for cl_num in cl_nums:
                cl_path = testpath+cl_num+"\\"
                sig_path = cl_path+sig

                if os.path.isfile( sig_path+"sig_testInfo.csv" ) == False:
                    #This Function : Adds column of grouped_t_no in the testInfo.csv   
                    #Create_Grouped_TNo_Column(cl_path)

                    # 1st Arg: The testInfo.csv file
                    # 2nd Arg: The SigPath where the new sig_csv should be pasted
                    MakeDir(sig_path)
                    sC = SignificantCluster(pd.read_csv( cl_path+"testInfo.csv" ), sig_path)
                    sC.Create_Sig_Cluster_DF()
                    del sC

                #sig_cl_df = pd.read_csv(sig_path+"sig_testInfo.csv")
                #tR = TestResults(sig_cl_df, sig_path, freq)
                # Uncomment the function which ever you would like to run:
                #tR.Composite_Images()
                #tR.Create_Histogram()
                #tR.TableOfClusterCompositeAndImagesInIt()
                #tR.Sil_Distribution( freq, cl_num )
                #tR.Remove_Pasted_Images()
                #del tR

def RunTestAnalysis( args ):
    #mLog = GetLogger(current_process().name)
    reg, freq = args
    path = "E:\\AllFrequencies\\"+reg+"\\"+freq+"\\"
    for fun in funList:
        funpath = path + fun[0] +"\\"
        AnalyzeEachClusterSizes(funpath, freq, fun[1])
        #CreateTableOfClusterComposite(funpath, fun[1])
        #CreateGraphicalComparisionBetweenClusterSize(funpath, fun[1])
        #break

def GetArguments():
    fList = ['19H','19V','19PCT','22V','37V','37H','37PCT','91H','91V','91PCT','150H','183_1H','183_3H','183_7H']
    rList = ['ATL','CPAC','EPAC','IO','SHEM','WPAC']
    arguments = []
    for r in rList:
        for f in fList:
            arguments.append( (r,f) )
    return arguments

if __name__ == '__main__':

    # arguments length will be equal to cpu_count()
    arguments = GetArguments()
    
    print("===================================================")
    s = time.time()
    # Processes according to total cores available
    print(cpu_count(),len(arguments))
    pool = Pool(processes=cpu_count())
    pool.map(RunTestAnalysis, arguments)
    pool.close()
    
    print("Total time taken : "+str( (time.time()-s)/60 )+" min")
    print("=================================================")
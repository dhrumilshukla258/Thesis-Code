from ast import literal_eval
from octant_creation import octa
from helper_function import GetLogger
from cluster_algorithms import Clustering
from test_data_selection import TestDataSelection

# Get current process details
from multiprocessing import current_process

import os
import cv2
import time
import numpy as np
import pandas as pd

class TestCases:
    def __init__(self):
        self.__mLog = GetLogger(current_process().name)
        
        self.testMethods = {}
        self.testMethods["agglo"] = pd.read_csv('..\\TestCases\\agglo.csv')
        self.testMethods["kmeans"] = pd.read_csv('..\\TestCases\\kmeans.csv')
        self.testMethods["fuzzykmeans"] = pd.read_csv('..\\TestCases\\fuzzykmeans.csv')
        for k,v in self.testMethods.items():
            self.testMethods[k].ResolutionX = self.testMethods[k].ResolutionX.astype(int)
            self.testMethods[k].ResolutionY = self.testMethods[k].ResolutionY.astype(int)
            self.testMethods[k].Channel = self.testMethods[k].Channel.astype(int)
            self.testMethods[k].ClusterSize = self.testMethods[k].ClusterSize.astype(int)
            self.testMethods[k].MaxIter = self.testMethods[k].MaxIter.astype(int)
            self.testMethods[k].RingSize = self.testMethods[k].RingSize.apply(literal_eval)
            
    def RunTestCase(self, k,_df, test, t_path):
        start_1 = time.time()
        tds = TestDataSelection(_df,
                                test.Area, 
                                (test.ResolutionX,test.ResolutionY),
                                test.Channel,
                                test.RingSize
                               )
        
        matrix, df = tds.StartSelection()
        
        cL = Clustering(matrix,
                        test.ClusterSize,
                        test.MaxIter
                       )
        
        end_1 = time.time()
        self.__mLog.debug("Time taken for Data Selection: " + str((end_1-start_1)/60) + " min")
        
        start_2 = time.time()
                
        if k == "kmeans":
            cL.Scikit_K_Means()
        elif k == "agglo":
            cL.Scipy_Agglomerative(test.DistanceMetric,
                                   test.Linkage,
                                   t_path
                                  )
        elif k == "fuzzykmeans":
            cL.Fuzzy_K_Means(test.ImageWeight,
                             test.PressureWeight,
                             test.WindWeight
                            )

        end_2 = time.time()

        self.__mLog.info("Average silhouette score for "+str(cL.k)+" cluster : "+str(cL.silhouetteAvg) )
        self.__mLog.debug("Time taken by Clustering Algorithm: " + str((end_2-start_2)/60) + " min")
        
        # Creating K cluster folders
        # Creating Images for according to cluster centers
        for i in range(int(cL.k)):
            if k=="fuzzykmeans" or k=="kmeans":
                img = cL.cluster_centers[i,0 : ( test.Channel * test.ResolutionX * test.ResolutionY ) ]
                
                if test.RingSize != 0:
                    octa.CreateImage(img, 
                                     test.ResolutionX,
                                     test.RingSize,
                                     test.Channel,
                                     t_path,
                                     i)
                else:
                    if test.Channel == 1:
                        img = img.reshape(test.ResolutionX,test.ResolutionY)
                    else:
                        img = img.reshape(test.ResolutionX,test.ResolutionY,3)
                    img = img.astype(np.uint8)
                    cv2.imwrite(t_path+"cluster_"+str(i)+".png",img)
            
            os.mkdir(t_path+str(i))
        
        
        #Create CSV with Image, ClusterLabel and Silhouette value
        image = pd.DataFrame({ 'FileName':df.FileName, 'ClusterLabel':cL.labels, 'SilhouetteVal':cL.silhouetteValues, 'T_No':df.T_No}) 
        image.to_csv(t_path+"testInfo.csv",index = False)
        
    def RunTests(self,_df,region,freq):
        for k,test in self.testMethods.items():
            path = "..//..//AllFrequencies//"+region+"//"+freq+"//"+k+"//"
            
            if os.path.isdir(path) == False:
                os.mkdir(path)

            for case in range(len(test)):
                if int(test.TestNo.iloc[case][5:]) >= 10 and int(test.TestNo.iloc[case][5:]) != 14:
                    start_1 = time.time()

                    # Creating a Test Directory
                    t_path = path + test.TestNo.iloc[case]
                    if os.path.isdir(t_path):
                        continue
                    os.mkdir(t_path)
                    t_path+="//"

                    self.__mLog.debug("Path for this Test Case:- "+t_path)

                    self.RunTestCase(k,_df,test.iloc[case],t_path)

                    self.__mLog.debug("Time taken -> "+region+" "+freq+" "+k+" "+test.TestNo.iloc[case]+" : "+str((time.time()-start_1)/60) + " min\n")
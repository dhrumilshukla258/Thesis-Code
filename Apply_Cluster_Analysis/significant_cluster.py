import numpy as np
import pandas as pd

class SignificantCluster():
    def __init__(self,df,path):
        self.__mDf = df
        self.__mPath = path
        self.__mSignificantCluster = self.__FindSignificantCluster()

    def __del__(self):
        #print(self.__mPath)    
        del self.__mDf
        del self.__mPath
        del self.__mSignificantCluster
    
    # Finding Significant Cluster by finding which intensity (t_no) passes the criteria
    def __FindSignificantCluster(self):
        t_no_label = self.__mDf.T_No.unique()
        t_no_label.sort()
        cluster_labels = self.__mDf.ClusterLabel.unique()
        cluster_labels.sort()
        
        # Length and Mean Sil of all T_No
        t_no_dict = {}
        for t_no in t_no_label:
            t_no_dict[t_no] = ( len(self.__mDf[ self.__mDf.T_No==t_no ]),np.mean( self.__mDf[ self.__mDf.T_No==t_no ].SilhouetteVal ) )
        
        significant_cluster_list = {}
        for cl in cluster_labels:
            # Get cl(1......k(clusters)) Cluster datafame and calculate its mean
            cl_mean = np.mean( self.__mDf[ self.__mDf.ClusterLabel==cl ].SilhouetteVal )
            
            for t_no in t_no_label:
                # Find Dataframe of Cluster and unique t_no
                # Gets total images found in T_No and CL
                cl_t_no_len = len( self.__mDf[(self.__mDf.ClusterLabel==cl) & (self.__mDf.T_No==t_no)] )
                
                # Total images found should be more than 5% of T_No images
                if cl_t_no_len > (0.05)*t_no_dict[t_no][0]:
                    
                    # Cluster sil value must be greater than T_No sil
                    if cl_mean > t_no_dict[t_no][1]:
                        if significant_cluster_list.get(cl) == None:
                            significant_cluster_list[cl] = []
                        significant_cluster_list[cl].append(t_no)
        return significant_cluster_list
    
    def Create_Sig_Cluster_DF(self):
        sig_list = []
        for cL,t_no_list in self.__mSignificantCluster.items():
            for t_no in t_no_list:
                #print(self.mDf[ (self.mDf.ClusterLabel == cL) & (self.mDf.T_No == t_no) ])
                sig_list.append( self.__mDf[ (self.__mDf.ClusterLabel == cL) & (self.__mDf.T_No == t_no) ] )
        sig_df = pd.concat(sig_list)
        sig_df.to_csv( self.__mPath+"sig_testInfo.csv", index=False,encoding='utf-8')

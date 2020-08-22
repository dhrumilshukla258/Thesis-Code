import os
import re
import cv2
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def MakeDir(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)
        return True
    return False

colorbar = {}
colorbar['19PCT']  = [270,320]
colorbar['19V'] = [185,295]
colorbar['19H'] = [110,290]
colorbar['22V'] = [205,290]
colorbar['37PCT']  = [250,310]
colorbar['37V'] = [205,290]
colorbar['37H'] = [135,290]
colorbar['91PCT']  = [135,300] 
colorbar['91V'] = [130,295]
colorbar['91H'] = [130,295]
colorbar['150H'] = [105,295]
colorbar['183_1H'] = [125,270]
colorbar['183_3H'] = [105,280]
colorbar['183_7H'] = [105,290]

class TestResults():
    global colorbar
    def __init__(self,df,path,freq):
        self.__mSigDf = df
        self.__mColorbarLimit = colorbar[freq]
        self.__mPath = path #+ "SilValue_"+ str(self.mMinSilVal) + "\\"
        
    def __del__(self):
        #print("--> " + self.__mPath)
        del self.__mSigDf
        del self.__mColorbarLimit
        del self.__mPath
        
    def Composite_Images(self,w=360):
        folderName = "Composite_Images"
        if not MakeDir(self.__mPath+folderName):
            return 
        folderName+="\\"

        cluster_labels = self.__mSigDf.ClusterLabel.unique()
        cluster_labels.sort()

        for cL in cluster_labels:
            cl_df = self.__mSigDf[ (self.__mSigDf.ClusterLabel == cL) & (self.__mSigDf.SilhouetteVal > 0)]
            
            totalLen = len(cl_df)
            comp_img = np.zeros((w,w,3))
            maxSilImage = 0
            maxSilVal = -2 # Min is -1
            
            for i,r in cl_df.iterrows():
                # Read ImageFiles and Not Considering Alpha Value just BGR is considered
                img_cv = cv2.imread(r.FileName) 
                img_cv = cv2.resize(img_cv, (w,w) )
                if r.SilhouetteVal > maxSilVal:
                    maxSilVal = r.SilhouetteVal
                    maxSilImage = img_cv
                comp_img += img_cv
            
            if totalLen != 0:
                comp_img /= totalLen
                self.Create_Image( maxSilImage, self.__mPath+folderName+str(cL)+"_MaxSil_Val_"+"{:.2f}".format(maxSilVal)+".png" ) 
                self.Create_Image( comp_img.astype(np.uint8), self.__mPath+folderName+str(cL)+"_"+str(totalLen)+".png" )
            
    def Create_Image(self,arr,path):
        plt.imshow(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB),cmap="jet_r")
        plt.clim(self.__mColorbarLimit[0],self.__mColorbarLimit[1])
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(path,bbox_inches = 'tight', pad_inches = 0)
        plt.close()
    
    '''        
    # Coping Images from MyCreatedData to The cluster where it belongs for this testCase
    def Paste_Images(self):
        for cL,t_no_list in self.__mSignificantCluster.items():
            MakeDir(self.__mPath+str(cL))
            
            for t_no in t_no_list:
                cl_t_no_df = self.__mDf[ (self.__mDf.ClusterLabel == cL) & (self.__mDf.T_No == t_no) ]
                if (len(cl_t_no_df) == len(self.__mDf)):
                    print("Somethings Wrong")
                    
                for i,r in cl_t_no_df.iterrows():
                    cL_path = self.__mPath+str(r.ClusterLabel)+"\\"
                    imgName = r.FileName.split("\\")[8]
                    
                    # Original Image Name in new path
                    actualFile = cL_path+imgName
                    
                    # Copied at the cluster label where it belongs
                    shutil.copy( r.FileName,cL_path )
                    
                    # Renaming the copied file : This helps us to understand which intensity belongs to a particular image
                    # Old name : imgName
                    # New name : imgIntensity_imgName
                    temp_t_no= str(t_no).replace(".","-")
                    renameFile = cL_path+temp_t_no+"_"+imgName[:25]+"_"+"{:.2f}".format(r.SilhouetteVal)+".png"
                    try:
                        os.rename(actualFile,renameFile)
                    except WindowsError:
                        os.remove(renameFile)
                        os.rename(actualFile,renameFile) 
    '''
    def Remove_Pasted_Images(self):
        for root, dirs, files in os.walk(self.__mPath):
            if "Cluster_Label" in root or  "T_No_Label" in root or "Composite_Images" in root:
                #print(os.path.join(root))
                shutil.rmtree(os.path.join(root))   
            #"Composite_Images" in root or
            ''' 
            for name in files:
                if "SSMIS" in name and ".png" in name:
                    print(os.path.join(root, name))
                    #os.remove(os.path.join(root, name))
            '''
    
    def Create_Histogram(self):
        cL_path = self.__mPath+"Cluster_Label\\"
        if not MakeDir(cL_path):
            return
        
        tno_path = self.__mPath+"T_No_Label\\"
        if not MakeDir(tno_path):
            return

        cluster_labels = self.__mSigDf.ClusterLabel.unique()
        cluster_labels.sort()
        t_no_labels = self.__mSigDf.T_No.unique()
        t_no_labels.sort()

        distr = {}
        distr["cluster_img"] = {}
        distr["cluster_sil"] = {}
        distr["t_no_img"] = {}
        distr["t_no_sil"] = {}
        plot_var = {}
        plot_var["cluster_img"] = ["T_No - per cluster", 0,9, len,cL_path]
        plot_var["t_no_img"] = ["Cluster Label - per T_No", min(cluster_labels),max(cluster_labels), len,tno_path]
        plot_var["cluster_sil"] = ["Sil Value - per cluster", -1,1,np.mean,cL_path]
        plot_var["t_no_sil"] = ["Sil Value - per T_No", -1, 1,np.mean,tno_path]

        for cL in cluster_labels:
            distr["cluster_img"][cL] = []
            distr["cluster_sil"][cL] = []

        for t_no in t_no_labels:
            distr["t_no_img"][t_no] = []
            distr["t_no_sil"][t_no] = []
        
        for i,r in self.__mSigDf.iterrows():
            cL = r.ClusterLabel
            t_no = r.T_No
            sil_val = r.SilhouetteVal
            distr["cluster_img"][cL].append( t_no )
            distr["t_no_img"][t_no].append( cL )
            distr["cluster_sil"][cL].append(sil_val)
            distr["t_no_sil"][t_no].append(sil_val)

        for label_name, v1 in distr.items():
            for number, lst in v1.items():
                plt.xlim( plot_var[label_name][1], plot_var[label_name][2] )
                plt.xlabel( plot_var[label_name][0] )
                plt.ylabel( "Img Distribution" )
                plt.hist(lst)

                val = "{:.2f}".format(plot_var[label_name][3](lst))
                plt.savefig(plot_var[label_name][4]+label_name[len(label_name)-3:]+"_"+str(number)+"_"+val+".png")
                plt.close()
        
        del distr
        del plot_var       
    
    def Sil_Distribution(self,freq,cl_size):
        if os.path.isfile(self.__mPath+"sil_distr.png"):
            return

        msg = freq+" "+cl_size+ "\n"
        msg += " Mean SilhouetteVal : "+str(np.mean(self.__mSigDf.SilhouetteVal)) + "\n"
        msg += " Total Images : "+ str(len(self.__mSigDf)) 
        #for t_no in t_no_labels:
        #msg += "Mean T_No Bin " + str(t_no) + " : " + str(np.mean( df.SilhouetteVal[ df.T_No == t_no ] ) ) + "\n"
        fig = plt.figure()
        plt.hist(self.__mSigDf.SilhouetteVal)
        plt.xlabel( "Sil Values" )
        plt.ylabel( "Img Distribution" )
        plt.xlim(-1,1)
        fig.suptitle(msg,fontsize=10)
        plt.savefig( self.__mPath+"sil_distr.png" )
        plt.close()



'''
def FindCommonTests():
    global regList
    global funList
    global freqList
    global imgFreq
    
    commonTest={}
    for reg in regList:
        #path = "..\\..\\AllFrequencies\\"+reg+"\\TestCases\\"
        #MakeDir(path)
        commonTest[reg] = {}
        for fun in funList:
            #funpath = path + fun + "\\"
            #MakeDir(funpath)
            commonTest[reg][fun] = {}
            for freq in freqList:
                for test,df in imgFreq[reg][freq][fun].items():
                    if commonTest[reg][fun].get(test) == None:
                        commonTest[reg][fun][test] = []    
                    commonTest[reg][fun][test].append(freq)
    return commonTest

def Create_Collage_For_AllFreq_Per_Unique_TestCases(freqList,reg,fun,test):
    paste_path = "..\\..\\AllFrequencies\\"+reg+"\\TestCases\\"+fun+"\\"
    original_path = "..\\..\\AllFrequencies\\"+reg+"\\"
    imgArr = []
    
    if len(freqList) != 14:
        return
    
    for freq in freqList:
        imgArr.append( cv2.imread( original_path+freq+"\\"+fun+"\\"+test+"\\sil_distr.png") )
    
    lastArr = []
    lastArr.append( imgArr[12] )
    lastArr.append( imgArr[13] )
    lastArr.append( np.zeros(imgArr[13].shape)+255 )
    lastArr.append( np.zeros(imgArr[13].shape)+255 )
    
    vis1 = np.concatenate(imgArr[:4], axis=1)
    vis2 = np.concatenate(imgArr[4:8], axis=1)
    vis3 = np.concatenate(imgArr[8:12], axis=1)
    vis4 = np.concatenate(lastArr,axis=1 )

    vis = np.concatenate( (vis1,vis2,vis3,vis4), axis=0 )
    cv2.imwrite(paste_path+test+".png", vis)

commonTest = FindCommonTests()
for reg,v1 in commonTest.items():
    for fun,v2 in v1.items():
        for test, freqList in v2.items():
            Create_Collage_For_AllFreq_Per_Unique_TestCases(freqList,reg,fun,test)

ls = []
for reg in regList:
    for freq in freqList:
        for fun in funList:
            for test,df in imgFreq[reg][freq][fun].items():
                path = "..\\..\\AllFrequencies\\"+reg+"\\"+freq+"\\"+fun+"\\"+test+"\\"
                
                if maxSil < np.mean( df.SilhouetteVal ):
                    maxSil = np.mean( df.SilhouetteVal )
                    best_sil_test = test

                if maxImg < len(df):
                    maxImg = len(df)
                    best_img_test = test

            ls.append( [ reg,freq,fun,maxSil,best_sil_test,maxImg, best_img_test] )
                #tR = TestResults(df,path,freq)
                #tR.Create_Histogram()
                #tR.Sil_Distribution(freq,test)
                #tR.Composite_Images()
                #tR.Remove_Pasted_Images()
                #tR.Paste_Images()
                #del tR
          
ls_df = pd.DataFrame(ls, columns=["Region", "Freq", "Method", "Max_SilValue", "Sil_TestCase", "Max_Img", "Img_TestCase"]  )
ls_df.to_csv( "temp.csv", index=False )
'''

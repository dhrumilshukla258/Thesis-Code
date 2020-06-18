import os
import re
import cv2
import copy
import json
import time
import numpy as np
import pandas as pd

# For Parallizing the Reading of Images
from multiprocessing import Pool, cpu_count

# Finding the Dvorak T_No based on wind speed      
def Find_TNo(wind):
    if (wind<25):
        return 0.0
    elif (wind>=25) and (wind<30):
        return 1.0
    elif (wind>=30) and (wind<35):
        return 2.0
    elif (wind>=35) and (wind<45):
        return 2.5
    elif (wind>=45) and (wind<55):
        return 3.0
    elif (wind>=55) and (wind<65):
        return 3.5
    elif (wind>=65) and (wind<77):
        return 4.0
    elif (wind>=77) and (wind<90):
        return 4.5
    elif (wind>=90) and (wind<102):
        return 5.0
    elif (wind>=102) and (wind<115):
        return 5.5
    elif (wind>=115) and (wind<127):
        return 6.0            
    elif (wind>=127) and (wind<140):
        return 6.5
    elif (wind>=140) and (wind<155):
        return 7.0
    elif (wind>=155) and (wind<170):
        return 7.5
    elif (wind>=170) and (wind<185):
        return 8.0
    else:
        return 8.5

# Finding Area in terms of Colored Image
def Find_Image_Area(imgPath):
    img = cv2.imread(imgPath,0)
    white_area = np.count_nonzero(img == 255)
    black_area = np.count_nonzero(img == 0)
    colored_area = 1 - ( (white_area+black_area)/(img.shape[0]*img.shape[1]) )
    return colored_area

class CreateCSV():
    def __init__(self, reg_freq_list, path = "..\\..\\MyCreatedData_New" ):
        # Initial Path
        self.__mPath = path
    
        # Arguments_Dict
        self.mResultRegFreq = {}
        
        for r,f in reg_freq_list:
            self.mResultRegFreq[(r,f)] = []
        
    def StartProcess(self):
        # Iterate over each image in MyCreatedData
        for root, dirs, files in os.walk(self.__mPath, topdown=False):
            
            if len(files) == 0:
                continue
            if ".png" not in files[0]:
                continue
            
            m = [m.start() for m in re.finditer(r'\\', root)]
            
            year = root[ m[2]+1 : m[3] ]
            region = root[ m[3]+1 : m[4] ]
            stormNo = root[ m[4]+1 : m[5] ]
            f_ =  root[ m[5]+1 : m[6] ]
            freq  = root[ m[6]+1 : ]
            
            if self.mResultRegFreq.get((region,freq)) != None:
                self.__ReadImage( year, region, stormNo, f_, freq, files )
                
        self.__CreatingCSV()
        
    def __ReadImage(self, year, region, stormNo, f_, freq, files ):
        
        storm_path = self.__mPath+"\\"+year+"\\"+region+"\\"+stormNo+"\\"
        freq_path = storm_path+f_+"\\"+freq+"\\"
        
        storm_data = pd.read_csv(storm_path+"StormData.txt",sep="\t")
        
        for f in files:
            filename = f[:len(f)-18]+".png"
            row = storm_data.loc[storm_data['FileName'] == filename]
            if len(row) == 1:
                imgPath = freq_path+f
                cenLon = row.iloc[0,1]
                cenLat = row.iloc[0,2]
                pressure = row.iloc[0,3]
                wind = row.iloc[0,4]
                area = Find_Image_Area(imgPath)
                t_no = Find_TNo(wind)
                self.mResultRegFreq[(region,freq)].append([ imgPath,cenLon,cenLat,pressure,wind,area,t_no ]) 
            else:
                print("Error No StormData : "+root+"\\"+f)
    
    def __CreatingCSV(self):
        # Store into excel file
        for (r,f),lst in self.mResultRegFreq.items():
            t = pd.DataFrame( lst )
            t.columns = ["FileName", "CenLon", "CenLat", "Pressure","Wind","Area","T_No"]
            t.to_csv("..\\ImagesPerFreq\\"+r+"_"+f+'.csv',index=False)
 
def DivideDataForMultipleProcess():
    fList = ['19H','19V','19','22V','37V','37H','37','91H','91V','91','150H','183_1H','183_3H','183_7H']
    rList = ['ATL','CPAC','EPAC','IO','SHEM','WPAC']
    
    # Here we could have used multiprocess manager but thats slow
    # so here is another trick which is to distribute data equally to all processes
    reg_freq_dict = {}
    
    #Dividing data based on total number of processes == if total cores are 4 
    #we get 21 (reg,freq) combo allocated to each processes
    total_allocation_of_argument_on_each_process = len(rList)*len(fList) / cpu_count()
    i = 0
    j = 0
    for r in rList:
        for f in fList:
            if reg_freq_dict.get(j) == None:
                reg_freq_dict[j] = []
            reg_freq_dict[ j ].append( (r,f) )
            
            i+=1
            if i > total_allocation_of_argument_on_each_process:
                i = 0
                j+=1
    
    # variable j should be equal to cpu_count()
    arguments = []
    for i,arg in reg_freq_dict.items():
        arguments.append(arg)
    
    return arguments

def RunProcess(args):
    obj = CreateCSV( args )
    obj.StartProcess()

if __name__ == '__main__':
 
    # arguments length will be equal to cpu_count()
    arguments = DivideDataForMultipleProcess()
     
    print("===================================================")
    s = time.time()
    
    # Processes according to total cores available
    pool = Pool(processes=cpu_count())
    pool.map(RunProcess, arguments)
    pool.close()
    
    print("Total time taken : "+str( (time.time()-s)/60 )+" min")
    print("=================================================")
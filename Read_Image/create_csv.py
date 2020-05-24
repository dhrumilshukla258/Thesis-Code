import os
import re
import cv2
import json
import numpy as np
import pandas as pd

# Creating Dict of CSVs
imgFreq = {}
fList = ['19H','19V','19','22V','37V','37H','37','91H','91V','91','150H','183_1H','183_3H','183_7H']
rList = ['ATL','CPAC','EPAC','IO','SHEM','WPAC']
for r in rList:
    imgFreq[r] = {}
    for f in fList:
        imgFreq[r][f] = []

# Finding the Dvorak T_No based on wind speed      
def Find_TNo(wind):
    if (wind>=25) and (wind<30):
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
 
# Iterate over each image in MyCreatedData
for root, dirs, files in os.walk("..\\..\\MyCreatedData", topdown=False):
    if len(files) == 0:
        continue
        
    if ".png" not in files[0]:
        continue
    
    m = [m.start() for m in re.finditer(r'\\', root)]
    
    region = root[ m[2]+1 : m[3] ]
    year = root[ m[3]+1 : m[4] ]
    stormNo = root[ m[4]+1 : m[5] ]
    f_ =  root[ m[5]+1 : m[6] ]
    freq  = root[ m[6]+1 : ]
    
    #print(year,region,stormNo,f_,freq)
    
    storm_data = pd.read_csv("..\\..\\MyCreatedData\\"+region+"\\"+year+"\\"+stormNo+"\\StormData.txt",sep="\t")
    
    for f in files:
        row = storm_data.loc[storm_data['FileName'] == f]
        if len(row) == 1:
            imgPath = root+"\\"+f
            cenLon = row.iloc[0,1]
            cenLat = row.iloc[0,2]
            pressure = row.iloc[0,3]
            wind = row.iloc[0,4]
            area = Find_Image_Area(imgPath)
            t_no = Find_TNo(wind)
            imgFreq[region][freq].append([ imgPath,cenLon,cenLat,pressure,wind,area,t_no ]) 
        else:
            print("Error No StormData : "+root+"\\"+f)

# Store into excel file
for r in rList:
    for f in fList:
        t = pd.DataFrame(imgFreq[r][f])
        t.columns = ["FileName", "CenLon", "CenLat", "Pressure","Wind","Area","T_No"]
        t.to_csv("..\\ImagesPerFreq\\"+r+"_"+f+'.csv',index=False)
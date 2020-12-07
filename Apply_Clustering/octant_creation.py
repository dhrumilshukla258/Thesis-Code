import os
import cv2
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from helper_function import GetLogger

# Get current process details
from multiprocessing import current_process

class Octants():
    def __init__(self):
        self.__mLog = GetLogger(current_process().name)
        self.o = {}
        
    def Calculate(self, w, ringsize):
        if self.o.get(str(w)+"_"+str(ringsize)) != None:
            return copy.deepcopy(self.o[str(w)+"_"+str(ringsize)])
        self.__mLog.debug("Calculated Octant")
        #width and height always same
        h=w
        arr = np.zeros((w,h),dtype=np.object)
        sectionArr = np.zeros((w,h), dtype=int)
        for i in range(w):
            for j in range(h):
                arr[i][j] = ( i-int(w/2), j-int(h/2) )
        
        # Creating Sections : We will add all the indices i.e ( (0,0) (1,0).....(n,m) ) to where they belong
        # in the section
        totalRing = len(ringsize)
        section = {}
        for i in range(int(totalRing)):
            section[i] = {}
            for j in range(8):
                section[i][j] = []            
        
        
        t1 = int(w/2)
        for k in range(1, int(totalRing+1) ):
            t2 = int ( ringsize[k-1]/2 )
            newArr = arr[  t1-t2:t1+t2, t1-t2:t1+t2 ]
            
            cenI = int ( newArr.shape[0]/2 )
            cenJ = int ( newArr.shape[1]/2 )
            preRing = int( ringsize[k-2]/2 )
            curRing = t2
            
            for i in range(newArr.shape[0]):
                for j in range(newArr.shape[1]):
                    
                    distance = math.sqrt( ( ( cenI-i)**2 ) + ( (cenJ-j)**2 ) )
                    
                    # Below are the edge cases : Same indices were added in 2 different sections
                    # To resolve I added these two condition
                    if k==totalRing:
                        if distance < preRing:
                            continue
                    elif k==1:
                        if distance>=curRing:
                            continue
                    else:
                        if distance < preRing or distance>=curRing:
                            continue
                    
                    # If above cases doesn't occur then the distance is such that its greater than
                    # previous Ring size and smaller than current Ring size
                    
                    # Once the circle is known we need to find the section where it belongs
                    # Angle helps in determing exactly where the indice belongs
                    angle = math.atan2( newArr[i][j][1], newArr[i][j][0] ) * 180 / np.pi
                    angle+=180
                    angle%=360

                    #Adding it in octants
                    if angle >= 0 and angle < 45:
                        section[k-1][0].append( ( newArr[i,j][0]+int(w/2), newArr[i,j][1]+int(h/2) ) )
                    elif angle >= 45 and angle < 90:
                        section[k-1][1].append( ( newArr[i,j][0]+int(w/2), newArr[i,j][1]+int(h/2) ) )
                    elif angle >= 90 and angle < 135:
                        section[k-1][2].append( ( newArr[i,j][0]+int(w/2), newArr[i,j][1]+int(h/2) ) )
                    elif angle >= 135 and angle < 180:
                        section[k-1][3].append( ( newArr[i,j][0]+int(w/2), newArr[i,j][1]+int(h/2) ) )
                    elif angle >= 180 and angle < 225:
                        section[k-1][4].append( ( newArr[i,j][0]+int(w/2), newArr[i,j][1]+int(h/2) ) )
                    elif angle >= 225 and angle < 270:
                        section[k-1][5].append( ( newArr[i,j][0]+int(w/2), newArr[i,j][1]+int(h/2) ) )
                    elif angle >= 270 and angle < 315:
                        section[k-1][6].append( ( newArr[i,j][0]+int(w/2), newArr[i,j][1]+int(h/2) ) )
                    elif angle >= 315 and angle < 360:
                        section[k-1][7].append( ( newArr[i,j][0]+int(w/2), newArr[i,j][1]+int(h/2) ) )
    
        
        # Assigning Each section a unique number
        app = {}
        totalSection = 0
        for i in range(totalRing):
            app[i] = {}
            for j in range(8):
                if app[i].get(j) == None:
                    app[i][j]=totalSection
                    totalSection+=1
        
        # Each Indices are given a section number where it belongs
        for i in range(totalRing):
            for j in range(8):
                for k in section[i][j]:
                    sectionArr[k[0],k[1]] = int(app[i][j])

        self.o[str(w)+"_"+str(ringsize)] = (sectionArr, totalSection )
        
        return copy.deepcopy(self.o[str(w)+"_"+str(ringsize)])
    
    def Draw(self,w,ringsize):
        if self.o.get(str(w)+"_"+str(ringsize)) != None:
            sectionArr, totalSection = self.o[str(w)+"_"+str(ringsize)]
        else:
            sectionArr, totalSection = self.Calculate( w, ringsize )
        
        #Code to print the maps of octants
        x = {}
        y = {}
        for i in range(w):
            for j in range(w):
                if x.get(sectionArr[i][j]) == None:
                    x[sectionArr[i][j]] = []
                if y.get(sectionArr[i][j]) == None:
                    y[sectionArr[i][j]] = []
                x[sectionArr[i][j]].append(i)
                y[sectionArr[i][j]].append(j)

        for k in range(totalSection):
            plt.scatter(x[k], y[k])
       
                
    def Parameters(self, matrix, w, ringsize, channel):
        if self.o.get(str(w)+"_"+str(ringsize)) != None:
            sectionArr, totalSection = copy.deepcopy(self.o[str(w)+"_"+str(ringsize)])
        else:
            sectionArr, totalSection = self.Calculate( w, ringsize )
        
        param = np.zeros(( totalSection*6*channel ))
        
        try:
            if channel == 1:
                mainMap = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY).flatten()
                sectionArr = sectionArr.flatten()
                
                t = 0
                for i in range(totalSection):
                    result = np.where(sectionArr == i)
                    miniArr = mainMap[result]
                    param[t] = np.subtract(*np.percentile(miniArr, [75, 25]))
                    param[t+1] = np.mean(miniArr)
                    param[t+2] = np.median(miniArr)
                    param[t+3] = np.std(miniArr) 
                    param[t+4] = np.min(miniArr)
                    param[t+5] = np.max(miniArr)
                    t+=6

            else:
                mainMap = matrix
                
                t = 0
                for i in range(totalSection):
                    result = np.where(sectionArr == i)
                    miniArr = mainMap[result]
                    for j in range(3):
                        param[t] = np.subtract(*np.percentile(miniArr[:,j], [75, 25]))
                        param[t+1] = np.mean(miniArr[:,j])
                        param[t+2] = np.median(miniArr[:,j])
                        param[t+3] = np.std(miniArr[:,j]) 
                        param[t+4] = np.min(miniArr[:,j])
                        param[t+5] = np.max(miniArr[:,j])
                        t+=6
        except:
            self.__mLog.debug("Error in Parameters Function of Octant Class: "+str(sys.exc_info()[0]))
        
        return param
    
    def CreateImage(self,param_matrix,w,ringsize,channel,t_path,clusterNo):
        if self.o.get(str(w)+"_"+str(ringsize)) != None:
            sectionArr, totalSection = copy.deepcopy(self.o[str(w)+"_"+str(ringsize)])
        else:
            sectionArr, totalSection = self.Calculate( w, ringsize )
        
        statArr = ["IQR", "Mean", "Median", "STD", "Min", "Max"]
        for stat in statArr:
            if os.path.isdir(t_path+stat) == False:
                os.mkdir(t_path+stat)
        
        try:
            if channel == 1:
                img = np.zeros((w,w))
                
                for i in range(6):
                    t = i
                    for j in range(totalSection):
                        result = np.where(sectionArr == j)
                        img[result] = param_matrix[t]
                        t+=6
                    img = img.astype(np.uint8)
                    cv2.imwrite(t_path+statArr[i]+"//cluster_"+str(clusterNo)+".png",img)
            
            else:
                img = np.zeros((w,w,channel))
                for i in range(6):
                    t = i
                    for j in range(totalSection):
                        result = np.where(sectionArr == j)
                        parameters = []
                        for k in range(3):
                            parameters.append(param_matrix[t])
                            t+=6
                        img[result] = parameters
                       
                    img = img.astype(np.uint8)
                    cv2.imwrite(t_path+statArr[i]+"//cluster_"+str(clusterNo)+".png",img)
        except:
            self.__mLog.debug("Error in CreateImage Function of Octant Class for "+t_path+" : "+str(sys.exc_info()[0]) )

octa = Octants()
octa.Calculate(360,[10,20,30,40,50,60,80,100,120,140,170,200,230,260,270,300,330,361])
octa.Calculate(120,[5,10,15,20,25,30,35,40,50,60,70,90,121])

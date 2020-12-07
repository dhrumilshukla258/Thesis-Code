from octant_creation import octa
from helper_function import GetLogger

# Get current process details
from multiprocessing import current_process

import numpy as np
import cv2

class TestDataSelection():
    def __init__( self, _df, area, resolution, channel, ringsize, freq, reg ):
        self.__mLog = GetLogger(current_process().name)
        
        # Consider area according to criteria below
        self.df = _df.loc[_df['Area'] >= area]
        
        self.w = resolution[0]
        self.h = resolution[1]
        self.channel = channel
        self.npmatrixfile = "OctantMatrix_"+reg+"_"+freq+"_"+str(area)+"_"+str(self.w)+"_"+str(channel)+".npy"
        self.ringsize = ringsize
        
    def StartSelection(self):
        if self.ringsize != 0:
            files = [f for f in os.listdir('NumpyFiles\\')]
            if self.npmatrixfile in files:
                return np.load("NumpyFiles\\"+self.npmatrixfile), self.df
            return self.__Octant()
        else:
            return self.__AllPixel()
    
    def __Octant(self):
        self.__mLog.debug("Checked Octant : "+ self.npmatrixfile[13:24])
        section, totalSection = octa.Calculate(self.w, self.ringsize)
        
        # Here instead of considering all pixels we consider statistical value of each sections
        # 8 is the image divided in octants
        # 6 is total statistical values: mean, median, maximum, minimum, standard deviation 
        # and interquartile range for each section
        matOfImages = np.zeros( ( len(self.df), ( self.channel * totalSection * 6 ) ) )
        
        i = 0
        for index, row in self.df.iterrows():
            imgName = row.FileName

            # Read ImageFiles and Not Considering Alpha Value just BGR is considered
            try:
                img_cv = cv2.imread(imgName) 
            except:
                self.__mLog.warning("Error in reading Image: "+imgName)
            
            try:
                img_cv = cv2.resize(img_cv, (self.w, self.h) )
            except:
                self.__mLog.warning("Resize Image Error: "+imgName)
            matOfImages[i] = octa.Parameters(img_cv, self.w, self.ringsize, self.channel)
            i+=1
        np.save("NumpyFiles\\"+self.npmatrixfile,matOfImages)
        return matOfImages, self.df
    
    def __AllPixel(self):
        # Creating Matrix of Images, each image considers all pixels
        # +2 removed
        matOfImages = np.zeros( ( len(self.df), ( self.channel * self.w * self.h ) ) )

        i = 0
        for index, row in self.df.iterrows():
            imgName = row.FileName

            # Read ImageFiles and Not Considering Alpha Value just BGR is considered
            try:
                img_cv = cv2.imread(imgName) 
            except NameError:
                self.__mLog.warning("Error in reading Image: "+imgName)
            
            try:
                img_cv = cv2.resize(img_cv, (self.w, self.h) )
            except TypeError:
                self.__mLog.warning("Resize Error: "+imgName)
                
            if self.channel == 1:
                mainMap = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).flatten()
            else:
                mainMap = img_cv.flatten()

            #alldata = np.append(mainMap, [row[3],row[4]], axis=0)
            matOfImages[i] = mainMap
            i+=1
        
        return matOfImages, self.df
import sys
sys.path.append('..\\Helper_Class')
#import os
#os.environ['PROJ_LIB'] = 'C:/ProgramData/Anaconda3/Lib/site-packages/mpl_toolkits/basemap'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from mpl_toolkits.basemap import Basemap, cm
import logging
logging.basicConfig(filename='..\\LogFile\\logFile_storm',level=logging.DEBUG)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

import scipy.io
import math
import cv2

from helper_functions import GetSerialDateFromString
from helper_functions import GetSerialDate
from helper_functions import GetNormalDate
from helper_functions import GetLocation
from scipy import interpolate

def WorldMap(lon,lat,loncorners,latcorners):
    # Plot the figure, define the geographic bounds
    fig = plt.figure(figsize=(6.67,5.0),dpi=72)

    m = Basemap(projection='cyl',
                lon_0 = lon,
                lat_0 = lat,
                llcrnrlat=latcorners[0],
                urcrnrlat=latcorners[1],
                llcrnrlon=loncorners[0],
                urcrnrlon=loncorners[1])

    m.drawcoastlines()
    m.drawmapboundary()
    
    '''        
    #Draw Parallel lines in the map
    parallels = np.arange(-60.,61,20.)
    m.drawparallels(parallels,labels=[True,False,True,False])

    #Draw Vertical lines in the map
    meridians = np.arange(-180.,180.,60.)
    m.drawmeridians(meridians,labels=[False,False,False,True])
    '''
    
    return fig, m

class StormReader:
    def __init__(self, storm, year, region, stormNo):
        self.mImagesWithProbability = {}
        
        #Stores the Storm dictionary
        self.__mStorm = storm

        #ShouldVisitFiles
        self.__goIn = True
        
        #Unique for each Storm
        self.__mInitialDir = "..\\..\\MyCreatedData\\"+str(year)+"\\"+region+"\\"+str(stormNo)+"\\"
        self.mYear = year
        self.mRegion = region
        self.mStormNo = stormNo
        self.__mImageDir = ""
        self.__mRootDirOfMatFile = ""
        self.__mMatFiles = ""
        self.__mStormBestTrack = 0
        
        #Unique for each file in the Storm folder
        self.__mLat = 0
        self.__mLon = 0
        self.__mDate = 0
        self.__mDateFile = 0
        self.__mBestTrack = 0
        self.__mBestTrackDate = 0

        self.__ReadBestTrack()
        
    def __ReadBestTrack(self):
        bestTrack = self.__mStorm["BestTrack"]
        '''
        bestTrack[0] provides root Directory
        bestTrack[1] filename
        Consist unedited trackfile, edited trackfile and Lightning Data
        '''
        reducedFile = ""
        for fi in bestTrack[1]:
            if "Reduced_Trackfile" in fi:
                reducedFile = fi
                break

        if reducedFile == "":
            self.__goIn = False
            return
        
        filename = "..\\" + bestTrack[0] + reducedFile
        
        self.__mBestTrack = pd.read_csv( filename,header=None,delim_whitespace=True )
        self.__mBestTrack = self.__mBestTrack.rename(columns={ 0:"year",1:"month",2:"day",3:"hour",4:"lat",5:"lon",6:"pressure", 7:"windspeed" } )
        self.__mBestTrack["date"] = pd.to_datetime( self.__mBestTrack[["year","month","day","hour"]])
        self.__mBestTrack = self.__mBestTrack.drop(columns=["year","month","day","hour"] )
        self.__mBestTrack = self.__mBestTrack[["date","lat","lon","pressure","windspeed",8]]
        
        self.__mBestTrackDate = GetSerialDate( self.__mBestTrack["date"] ) - GetSerialDate( self.__mBestTrack["date"][0] )
  
    def ReadStorm(self):
        if self.__goIn == False:
            return
        
        self.__mStormBestTrack = pd.DataFrame( columns= ['FileName','Lon','Lat','Pressure','Windspeed'])
        
        for f_,files in self.__mStorm.items():
            if f_ == "BestTrack":
                continue
            
            #files[0] means root dir where all mat files are
            #files[1] means set of mat files for satellite f_16/17/18/19
            self.__mRootDirOfMatFile = "..\\" + files[0]
            self.__mMatFiles = files[1]
            self.__mImageDir = self.__mInitialDir + f_ + "\\"
            
            self.mImagesWithProbability[f_] = {}
            self.mImagesWithProbability[f_]["S1"] = {}
            self.mImagesWithProbability[f_]["S2"] = {}
            self.mImagesWithProbability[f_]["S3"] = {}
            self.mImagesWithProbability[f_]["S4"] = {}
            
            self.mImagesWithProbability[f_]["S1"]["19V"] = []
            self.mImagesWithProbability[f_]["S1"]["19H"] = []
            self.mImagesWithProbability[f_]["S1"]["22V"] = []
            
            self.mImagesWithProbability[f_]["S2"]["37V"] = []
            self.mImagesWithProbability[f_]["S2"]["37H"] = []

            self.mImagesWithProbability[f_]["S3"]["150H"] = []
            self.mImagesWithProbability[f_]["S3"]["183_1H"] = []
            self.mImagesWithProbability[f_]["S3"]["183_3H"] = []
            self.mImagesWithProbability[f_]["S3"]["183_7H"] = []

            self.mImagesWithProbability[f_]["S4"]["91V"] = []
            self.mImagesWithProbability[f_]["S4"]["91H"] = []

            self.__ReadAllMatFile(f_)
        self.__mStormBestTrack.to_csv( self.__mInitialDir+"StormData.txt", sep='\t', index=False,encoding='utf-8')


    def __ReadAllMatFile(self,f_):
        for filename in self.__mMatFiles:

            self.__ReadMatFile(filename,f_)

    def __ReadMatFile(self,filename,f_):
        try:
            mat = scipy.io.loadmat(self.__mRootDirOfMatFile + filename)
        except:
            logging.debug("Error Reading File: " + str( self.__mRootDirOfMatFile + filename ) )
            return
        
        swaths = mat["passData"][0][0]
    
        mSwathMap = {}
        mSwathMap[1] = [ "19V", "19H", "22V" ]
        mSwathMap[2] = [ "37V", "37H" ]
        mSwathMap[3] = [ "150H", "183_1H", "183_3H", "183_7H" ]
        mSwathMap[4] = [ "91V", "91H" ]

        i = 1
        for swath_data in swaths:
            #In function num2date which we will use to convert 
            #Serial date to Normal Date the date starts with
            #Jan 1,0001 Hence subtracting -366 to increase 1 year
            serial_date = swath_data[0][0][0] - 366
            self.__mDate = GetNormalDate(serial_date)
            self.__mDateFile = GetSerialDateFromString(filename[:15])
            self.__mLat = swath_data[0][0][1]
            self.__mLon = swath_data[0][0][2]
            channels = swath_data[0][0][3]
            
            #If any of the Lat and Lon in the file are invalid
            if self.__mLat.any() >= 91 or self.__mLat.any() <= -91 or self.__mLon.any() >= 181 or self.__mLon.any() <= -181:
                continue
            
            centerLon, centerLat, pressure, wind  = self.__CenterPositionAndWindPressure()
            
            if centerLon == 181 or centerLat == 91:
                continue
            
            #Store Data of a .mat file
            self.__mStormBestTrack = self.__mStormBestTrack.append( {'FileName' : filename[ : len(filename) - 4]+".png",
                                                                        'Lon'      : centerLon,
                                                                        'Lat'      : centerLat,
                                                                        'Pressure' : pressure,
                                                                        'Windspeed': wind }, ignore_index=True)
            
            #Finding corners of map by Great Circle Distance
            a = math.inf
            b = -math.inf
            c = math.inf
            d = -math.inf
            for loop in range(0,360):
                t1,t2 = GetLocation(centerLon, centerLat,loop, 400)
                a = min(t1,a)
                b = max(t1,b)
                c = min(t2,c)
                d = max(t2,d)
            loncorners = ([a,b])
            latcorners = ([c,d])
            
            j = 0
            shouldDraw = True
            for freq in channels:
                if shouldDraw: 
                    if self.__CheckCriteria(freq) == True:
                        image_file_dir = self.__mImageDir + "S" + str(i) + "\\" + mSwathMap[i][j] + "\\" + filename[ : len(filename) - 4]
                    
                        #Creating Image
                        fig, m = WorldMap(centerLon, centerLat,loncorners,latcorners)
                        a,b = m(self.__mLon,self.__mLat)
                        plt.pcolormesh(a,b,freq, cmap="jet_r")

                        #Checking if All point are in ocean
                        if j==0:
                            for tempI in range(self.__mLon.shape[0]):
                                if shouldDraw:
                                    for tempJ in range(self.__mLon.shape[1]):
                                        if m.is_land(self.__mLon[tempI,tempJ],self.__mLat[tempI,tempJ]):
                                            shouldDraw = False
                                            break
                        
                        if shouldDraw:
                            plt.tight_layout(pad=0)
                            plt.savefig(image_file_dir,bbox_inches = 'tight', pad_inches = 0)
                            
                            #Check Probability of how much area is covered by whitespace
                            imageData = [ filename[ : len(filename) - 4]+".png" , self.__ProbabilityOfColorImage(image_file_dir+".png") ]
                            self.mImagesWithProbability[f_]["S" + str(i)][mSwathMap[i][j]].append( imageData )
                        
                        plt.close( fig )
                else:
                    break
                j+=1                          
            i+=1

    def __CenterPositionAndWindPressure(self):
        x = self.__mBestTrackDate
        ylat = self.__mBestTrack["lat"]
        ylon = self.__mBestTrack["lon"]
        
        xnew = self.__mDateFile - GetSerialDate( self.__mBestTrack["date"][0] )
  
        newlat=91
        newlon=181
        
        #Not to consider value before and after the best track
        if xnew < 0 or xnew > x[len(x)-1]:
            return newlon,newlat, 0, 0

        
        if len(x) > 1:
            tck = interpolate.CubicSpline(x, ylat)
            newlat = tck(xnew)
            
            tck = interpolate.CubicSpline(x, ylon)
            newlon = tck(xnew)
        else:
            newlat = ylat
            newlon = ylon

        closest = math.inf
        rowNumber = -1
        for i in range(len(x)):
            if abs( x[i] - xnew ) < closest:
                closest = abs( x[i] - xnew )
                rowNumber = i
        
        return newlon,newlat, self.__mBestTrack["pressure"][rowNumber], self.__mBestTrack["windspeed"][rowNumber]
   
    def __CheckCriteria(self, freq):
        #If any brightness temperature hase value above 320
        #and below 0 then its invalid
        showDraw = True
        if freq.shape[0] == 0 or freq.shape[1] == 0:
            return False

        for row in freq:
            for col in row:
                if col > 320 or col<0:
                    showDraw = False
        return showDraw
        
    def __ProbabilityOfColorImage(self, imgName ):   
        #Check Probability of how much area is covered by whitespace
        img_cv = cv2.imread(imgName)  
        img_cv = cv2.resize(img_cv, (360,360))
        
        count = 0
        for i in range(img_cv.shape[0]):
            for j in range(img_cv.shape[1]):
                if ( img_cv[i][j][0] == 255 and img_cv[i][j][1] == 255 and img_cv[i][j][2] == 255):
                    count+=1
                    
        return 1 - ( count / (img_cv.shape[0]*img_cv.shape[1]) )

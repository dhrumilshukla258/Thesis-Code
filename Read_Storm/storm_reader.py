'''
Use class StormReader to iterate through matfiles found in the storm
It can calculate the BestTrack deatils 
i.e. Center Latitude and Longitude, Pressure and Wind
'''
import math
import time

# For reading and creating tables
import pandas as pd

# To Interpolate storm center lon and lat
from scipy import interpolate

from matfile_reader import MatReader

# Get current process details
from multiprocessing import current_process

# Helper Function
from helper_function import MakeDir
from helper_function import GetLogger
from helper_function import GetSerialDate
from helper_function import GetSerialDateFromString

class StormReader():
    def __init__(self,stormDict,region,year,stormNo):
        self.__mStormDict = stormDict
        self.__mYear = year
        self.__mRegion = region
        self.__mStormNo = stormNo
        
        # Setting the path where Images will be created
        path= "..\\..\\MyCreatedData_New\\"
        MakeDir(path+self.__mRegion)
        MakeDir(path+self.__mRegion+"\\"+self.__mYear)
        MakeDir(path+self.__mRegion+"\\"+self.__mYear+"\\"+self.__mStormNo)
        self.__mInitialDir = path+self.__mRegion+"\\"+self.__mYear+"\\"+self.__mStormNo+"\\"
        self.__mSatelliteDir = 0
        
        # Unique for each storm
        self.__mRootDirOfMatFile = 0
        self.__mMatFiles = 0
        self.__mBestTrack = 0
        self.__mBestTrackDate = 0
        
        # Unique for each satellite
        self.__mMatReaderClass = 0
        
        # Logger
        self.__mLog = 0
        
    def __HasBestTrack(self):
        bestTrack = self.__mStormDict["BestTrack"]

        reducedFile = ""
        for fi in bestTrack[1]:
            if "Reduced_Trackfile" in fi:
                reducedFile = fi
                break

        if reducedFile == "":
            print("Reduced_Trackfile not found :",region,year,stormNo)
            return False
        
        filename = "..\\" + bestTrack[0] + reducedFile
        self.__mBestTrack = pd.read_csv( filename,header=None,delim_whitespace=True )
        self.__mBestTrack = self.__mBestTrack.rename(columns={ 0:"year",1:"month",2:"day",3:"hour",4:"lat",5:"lon",6:"pressure", 7:"windspeed" } )
        self.__mBestTrack["date"] = pd.to_datetime( self.__mBestTrack[["year","month","day","hour"]])
        self.__mBestTrack = self.__mBestTrack.drop(columns=["year","month","day","hour"] )
        self.__mBestTrack = self.__mBestTrack[["date","lat","lon","pressure","windspeed",8]] 
        self.__mBestTrackDate = GetSerialDate( self.__mBestTrack["date"] ) - GetSerialDate( self.__mBestTrack["date"][0] )
    
        return True
    
        
    def ReadStorm(self):
        start = time.time()
        self.__mLog = GetLogger(current_process().name)
        if self.__HasBestTrack() == False:
            msg = "Total time Reading: " + self.__mYear + " " + self.__mRegion + " " + self.__mStormNo + ": " + str( (time.time()-start)/60 ) + " minutes"
            self.__mLog.debug("===========================================================")
            self.__mLog.debug(msg)
            self.__mLog.debug("===========================================================")
            return
        
        self.__mStormBestTrack = pd.DataFrame( columns= ['FileName','Lon','Lat','Pressure','Windspeed'])
        
        # Iterate over all the Satellites in this storm
        for f_,files in self.__mStormDict.items():
            if f_ == "BestTrack":
                continue
            
            MakeDir(self.__mInitialDir+f_)
            self.__mSatellitePath = self.__mInitialDir+f_+"\\"
            
            # files[0] : root dir where all mat files are
            # files[1] : set of mat files for satellite f_16/17/18/19
            self.__mRootDirOfMatFile = "..\\" + files[0]
            matFiles = files[1]
            
            self.__mMatReaderClass = MatReader(self.__mRootDirOfMatFile,self.__mSatellitePath)
            
            # Iterate over all Matfiles
            for filename in matFiles:
                self.__ReadMatFiles(filename)
            
        self.__mStormBestTrack.to_csv( self.__mInitialDir+"StormData.txt", sep='\t', index=False,encoding='utf-8')
        
        msg = "Total time Reading : " + self.__mYear + " " + self.__mRegion + " " + self.__mStormNo + ": " + str( (time.time()-start)/60 ) + " minutes"
        self.__mLog.debug("===========================================================")
        self.__mLog.debug(msg)
        self.__mLog.debug("===========================================================")

    def __ReadMatFiles(self,filename):
        matfile_date = GetSerialDateFromString(filename[:15])
        
        # Finding Necessary details from the BestTrack File
        centerLon, centerLat, pressure, wind  = self.__CenterPositionAndWindPressure(matfile_date)
        
        if centerLon == 181 or centerLat == 91:
            return
        
        # Store Data of a .mat file in a dataframe
        self.__mStormBestTrack = self.__mStormBestTrack.append( {'FileName' : filename[ : len(filename) - 4]+".png",
                                                                    'Lon'      : centerLon,
                                                                    'Lat'      : centerLat,
                                                                    'Pressure' : pressure,
                                                                    'Windspeed': wind }, ignore_index=True)
        # Reading the MatFile
        self.__mMatReaderClass.ReadFileAndCreateImages( filename, centerLon, centerLat )
        
    def __CenterPositionAndWindPressure(self,matfile_date):
        x = self.__mBestTrackDate
        ylat = self.__mBestTrack["lat"]
        ylon = self.__mBestTrack["lon"]
        
        xnew = matfile_date - GetSerialDate( self.__mBestTrack["date"][0] )
  
        newlat=91
        newlon=181
        
        # Not to consider value before and after the best track
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
'''
Use class MatReader to create images from a single matFile if
the required criteria is satisfied for each frequency images
will be created accordingly
'''
import numpy as np

# Reading Mat File
import scipy.io

# For Creating Images
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# To find Great Circle Distance
from cartopy import geodesic
import shapely.geometry as sgeom

# To Mask and Find Land and CoastLine
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from shapely.prepared import prep

# Get current process details
from multiprocessing import current_process

# Helper Function
from helper_function import MakeDir
from helper_function import GetLogger

land_shp_fname = shpreader.natural_earth(resolution='50m',
                                       category='physical', name='land')

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
land = prep(land_geom)

def IsLand(lon,lat):
    global land
    for x in lon:
        for y in lat:
            if land.contains(sgeom.Point(x,y)):
                return True
    return False
    
class MatReader():    
    def __init__(self,matpath,satpath):
        #self.mReg = reg
        self.__mLog = 0
        self.__mErrors = {}
        self.mMatPath = matpath
        self.mSatellitePath = satpath
        
        self.__mFreq_to_swath = {}
        self.__mFreq_to_swath['19']  = [0,[2.400, 1.400]] 
        self.__mFreq_to_swath['19V'] = [0,0]
        self.__mFreq_to_swath['19H'] = [0,1]
        self.__mFreq_to_swath['22V'] = [0,2]

        self.__mFreq_to_swath['37']  = [1,[2.150, 1.150]]
        self.__mFreq_to_swath['37V'] = [1,0]
        self.__mFreq_to_swath['37H'] = [1,1]

        self.__mFreq_to_swath['91']  = [3,[1.751, 0.751]]
        self.__mFreq_to_swath['91V'] = [3,0] 
        self.__mFreq_to_swath['91H'] = [3,1]

        self.__mFreq_to_swath['150H'] = [2,0]
        self.__mFreq_to_swath['183_1H'] = [2,1]
        self.__mFreq_to_swath['183_3H'] = [2,2]
        self.__mFreq_to_swath['183_7H'] = [2,3]
        
    def PCT_Function(self,v,h,val):
        return val[0]*v - val[1]*h
    
    def ReadFile(self,matFile,centerLon,centerLat):
        self.__mErrors[0] = ["Invalid Lat and Lon values     : ",[]]
        self.__mErrors[1] = ["Invalid shape of Frequency     : ",[]]
        self.__mErrors[2] = ["Invalid Brightness Temperature : ",[]]
        self.__mErrors[3] = ["Land found in smaller region   : ",[]]
        
        self.__mLog = GetLogger(current_process().name)
        try:
            mat = scipy.io.loadmat(self.mMatPath+matFile)
        except:
            msg = "Error Reading File: " + str( self.mMatPath+matFile ) 
            self.__mLog.error( msg )
        
        swaths = mat["passData"][0][0]
        
        for freq,swathList in self.__mFreq_to_swath.items():
            
            # Setting image path
            freqPath = self.mSatellitePath+"\\"+freq
            imgPath = freqPath+"\\"+matFile[:len(matFile)-4]+".png"
            
            # Reading Freqdata
            swath_data = swaths[ swathList[0] ]
            lat = swath_data[0][0][1]
            lon = swath_data[0][0][2]
            channel = swath_data[0][0][3]
            
            # Calculating PCT values for specific frequencies
            if freq == '19' or freq=='37' or freq=='91':
                tbs = self.PCT_Function( channel[0], channel[1], swathList[1] )
            else:
                tbs = channel[swathList[1]]
            
            goForward, msg = self.__CheckCriteria(lon,lat,tbs)
            if goForward == False:
                self.__mErrors[msg][1].append(freq)
                continue
            
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            # Axes (Lat and Lon)
            #ax.set_xticks(list(np.arange(-180,180,5)), crs=ccrs.PlateCarree())
            #ax.set_yticks(list(np.arange(-90,90,5)), crs=ccrs.PlateCarree())

            # Selecting some portion of the image by CenterLon and CenterLat
            circle= geodesic.Geodesic().circle(centerLon, centerLat, 400000)
            poly = sgeom.Polygon(circle)
            ax.set_extent( [poly.bounds[0], poly.bounds[2], poly.bounds[1], poly.bounds[3]] )
            
            # Another Check to see if the smaller area has land near it
            if IsLand( np.arange(poly.bounds[0],poly.bounds[2],0.1),np.arange(poly.bounds[1],poly.bounds[3],0.1)  ):
                plt.close()
                self.__mErrors[3][1].append(freq)
                continue
            
            # Creating Directory for Frequency
            MakeDir(freqPath)
            
            # Mapping Lon, Lat and Brightness Temperature
            pc = ax.pcolormesh(lon, lat, tbs, cmap="jet_r")
            
            # Mask Land and CostLines
            ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='black')
            ax.add_feature(cfeature.COASTLINE, edgecolor='black', facecolor='black')
            
            # Removes extra padding from the figure
            plt.tight_layout(pad=0)
            
            # To add colorbar
            #plt.colorbar(pc)
            
            # Storing the Image at the mentioned location
            plt.savefig(imgPath,bbox_inches = 'tight', pad_inches = 0)
            plt.close()
        
        for k,v in self.__mErrors.items():
            if len(v[1]) != 0:
                self.__mLog.warning(v[0]+matFile+" : "+str(v[1]))
    
    def __CheckCriteria(self, lon, lat, tbs):
        # If any of the Lat and Lon in the file are invalid
        if  np.any(lat>=91) or np.any(lat<=-91) or np.any(lon>= 181) or np.any(lon<=-181):
            return False, 0
        
        if tbs.shape[0] == 0 or tbs.shape[1] == 0:
            return False, 1
        
        # If any brightness temperature has value above 320
        # and below 0 then its invalid
        if np.any(tbs > 320) or np.any(tbs<0):
            return False, 2
        
        return True, 4
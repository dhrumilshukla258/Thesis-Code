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

freq_to_swath = {}
# Index 0 means swath number [ S1, S2, S3, S4 ]
# Index 1 for PCT Frequency is the equation value and for normal frequency i.e in S1/S2/S3/S4 where it belongs (index in that swath)
# Index 2 is the colormap range
# Index 3 is the validation range for the max_tbs is 400 for smaller freq and 320 for higher freq | min_tbs is >0 for all freq
freq_to_swath['19']  = [0,[2.400, 1.400],[270,320],400]
freq_to_swath['19V'] = [0,0,[185,295],400]
freq_to_swath['19H'] = [0,1,[110,290],400]
freq_to_swath['22V'] = [0,2,[205,290],400]

freq_to_swath['37']  = [1,[2.150, 1.150],[250,310],400]
freq_to_swath['37V'] = [1,0,[205,290],400]
freq_to_swath['37H'] = [1,1,[135,290],400]

freq_to_swath['91']  = [3,[1.751, 0.751],[135,300],320] 
freq_to_swath['91V'] = [3,0,[130,295],320]
freq_to_swath['91H'] = [3,1,[130,295],320]

freq_to_swath['150H'] = [2,0,[105,295],320]
freq_to_swath['183_1H'] = [2,1,[125,270],320]
freq_to_swath['183_3H'] = [2,2,[105,280],320]
freq_to_swath['183_7H'] = [2,3,[105,290],320]

def IsLand(lon,lat):
    global land
    for x in lon:
        for y in lat:
            if land.contains(sgeom.Point(x,y)):
                return True
    return False
    
class MatReader():
    global freq_to_swath
    def __init__(self,matpath,satpath):
        #self.mReg = reg
        self.__mLog = 0
        self.__mErrors = {}
        self.mMatPath = matpath
        self.mSatellitePath = satpath

    def PCT_Function(self,v,h,val):
        return val[0]*v - val[1]*h

    def ReadMatFile(self,matFile):
        try:
            mat = scipy.io.loadmat(self.mMatPath+matFile)
        except:
            msg = "Error Reading File: " + str( self.mMatPath+matFile ) 
            self.__mLog.error( msg )
        return mat

    def ReadFileAndCreateImages(self,matFile,centerLon,centerLat,valid_freq = ['19','19V','19H','22V','37','37V','37H','91','91V','91H','150H','183_1H','183_3H','183_7H']):
        self.__mErrors[0] = ["Invalid Lat and Lon values     : ",[]]
        self.__mErrors[1] = ["Invalid shape of Frequency     : ",[]]
        self.__mErrors[2] = ["Invalid Brightness Temperature : ",[]]
        self.__mErrors[3] = ["Land found in smaller region   : ",[]]
        
        self.__mLog = GetLogger(current_process().name)
        
        mat = self.ReadMatFile(matFile)
        swaths = mat["passData"][0][0]
        
        # Selecting some portion of the image by CenterLon and CenterLat
        circle= geodesic.Geodesic().circle(centerLon, centerLat, 400000)
        poly = sgeom.Polygon(circle)

        for freq in valid_freq:
            swathList = freq_to_swath[freq]
            
            # Setting image path
            freqPath = self.mSatellitePath+"\\"+freq
            
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
            
            goForward, msg = self.__CheckCriteria(lon,lat,tbs,swathList[3])
            if goForward == False:
                self.__mErrors[msg][1].append(freq)
                continue
            
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            # Axes (Lat and Lon)
            #ax.set_xticks(list(np.arange(-180,180,5)), crs=ccrs.PlateCarree())
            #ax.set_yticks(list(np.arange(-90,90,5)), crs=ccrs.PlateCarree())

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
            previous_clim = pc.get_clim()
            pc.set_clim(swathList[2][0], swathList[2][1])
        
            # Mask Land and CostLines
            ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='black')
            ax.add_feature(cfeature.COASTLINE, edgecolor='black', facecolor='black')
            
            # Removes extra padding from the figure
            plt.tight_layout(pad=0)
            
            # To add colorbar
            #plt.colorbar(pc)
            
            # Storing the Image at the mentioned location
            # Setting Image Path
            minTbs = "{:.2f}".format(previous_clim[0])
            maxTbs = "{:.2f}".format(previous_clim[1])
            imgPath = freqPath+"\\"+matFile[:len(matFile)-4]+"_"+minTbs+"_"+maxTbs+".png"
            plt.savefig(imgPath,bbox_inches = 'tight', pad_inches = 0)
            plt.close()
        
        for k,v in self.__mErrors.items():
            if len(v[1]) != 0:
                self.__mLog.warning(v[0]+matFile+" : "+str(v[1]))

    def __CheckCriteria(self, lon, lat, tbs, max_tbs):
        # If any of the Lat and Lon in the file are invalid
        if  np.any(lat>=91) or np.any(lat<=-91) or np.any(lon>= 181) or np.any(lon<=-181):
            return False, 0
        
        if tbs.shape[0] == 0 or tbs.shape[1] == 0:
            return False, 1
        
        # If any brightness temperature has value above 320
        # and below 0 then its invalid
        if np.any(tbs > max_tbs) or np.any(tbs<0):
            return False, 2
        
        return True, 4
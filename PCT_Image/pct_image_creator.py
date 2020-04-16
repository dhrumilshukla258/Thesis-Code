import re
import json
import copy
import time
import os.path
import logging
import pandas as pd


imgFreq = {}
fList = ['19H','19V','22V','37V','37H','91H','91V','150H','183_1H','183_3H','183_7H']
rList = ['ATL','CPAC','EPAC','IO','SHEM','WPAC']
for r in rList:
    imgFreq[r] = {}
    for f in fList:
        imgFreq[r][f] = pd.read_csv("..//ImagesPerFreq//"+r+"_"+f+'.csv')


pctlist = [['19H','19V','19'], 
           ['37H','37V','37'],
           ['91H','91V','91']]
s1 = time.time()

pctDf = {}
for reg in rList:
    pctDf[reg] = {}
    for freq in pctlist:
        
        df1 = imgFreq[reg][freq[0]]
        df2 = imgFreq[reg][freq[1]]
        
        fName1 = list(df1.FileName)
        fName2 = list(df2.FileName)
        fName = []

        s2 = time.time()
        ind = 0
        for f1 in fName1:
            
            m = [m.start() for m in re.finditer(r'//', f1)] 
            fName.append( ( f1,
                            f1[ :m[7] ] + "//" + freq[1] + f1[ m[8]: ],
                            f1[ :m[7] ] + "//" + freq[2] + f1[ m[8]: ],
                            df1.CenLon.iloc[ind],
                            df1.CenLat.iloc[ind],
                            df1.Pressure.iloc[ind],
                            df1.Wind.iloc[ind],
                            df1.Area.iloc[ind],
                            df1.T_No.iloc[ind] ) 
                        )
            ind+=1
            
        newFreq = []
        ind = 0
        for f in fName:
            if f[1] in fName2:
                newFreq.append(f)
                ind+=1
                
        pctDf[reg][freq[2]] = pd.DataFrame.from_records(newFreq, columns =['F1', 'F2', 'FileName','CenLon','CenLat','Pressure','Wind','Area','T_No'])

        e2=time.time()
        print(reg,freq,len(fName1),len(fName2),ind,e2-s2)

e1 = time.time()
print(e1-s1)

file_per_year = json.load(open('..//files_per_year.json'))

import math
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm

logging.basicConfig(filename='..//LogFile//logFile_pct_image',level=logging.DEBUG)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


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
    
    return fig, m

def GetLocation(lon1,lat1, brng, distanceKm):
    try:
        lat1 = lat1 * math.pi/ 180.0
        lon1 = lon1 * math.pi / 180.0
        #earth radius
        R = 6378.1
        #R = ~ 3959 MilesR = 3959

        distanceKm = distanceKm/R

        brng = (brng / 90)* math.pi / 2

        lat2 = math.asin(math.sin(lat1) * math.cos(distanceKm) 
        + math.cos(lat1) * math.sin(distanceKm) * math.cos(brng))

        lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(distanceKm)
        * math.cos(lat1),math.cos(distanceKm)-math.sin(lat1)*math.sin(lat2))

        lon2 = 180.0 * lon2/ math.pi
        lat2 = 180.0 * lat2/ math.pi
        return lon2,lat2
    except AssertionError as error:
        logging.debug(error)


swathFreq = {}
swathFreq['19'] = 0
swathFreq['37'] = 1
swathFreq['91'] = 3

def PCT_Function(v,h,val):
    return val[0]*v - val[1]*h

def ReadMatFile(fpath,freq):
    try:
        mat = scipy.io.loadmat(fpath)
    except:
        logging.debug("Error Reading File: " + str( fpath ) )
        return

    swaths = mat["passData"][0][0]
    currSwath = swaths[ swathFreq[freq[0]] ]
    lat = currSwath[0][0][1]
    lon = currSwath[0][0][2]
    channel = currSwath[0][0][3]
    
    #Here we consider channel[0] and channe[1] cause
    #19V, 37V and 91V all exists on channel[0] and
    #19H, 37H and 91H all exists on channel[1]
    diff = PCT_Function(channel[0],channel[1],(freq[1],freq[2]))
    
    return lon,lat,diff
    
def GetLatLonCorners(centerLon,centerLat):
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
    return loncorners, latcorners

pcteqn = [ ['19', 2.400, 1.400],
           ['37', 2.150, 1.150],
           ['91', 1.751, 0.751] ]


s1 = time.time()
for reg in rList:
    logging.debug("Images for "+reg+" region")
    logging.debug("=========================================================================")
    
    s2 = time.time()
    for freqArr in pcteqn:
        logging.debug("Frequency "+freqArr[0])
        logging.debug("===========================================")
        df = pctDf[reg][freqArr[0]]
        s3 = time.time()
        
        s4 = time.time()
        preYear = -1
        tot = 0
        for i,r in df.iterrows():
            #Reading V and H image in this loop
            m = [m.start() for m in re.finditer(r'//', r.F1)]
            year = r.F1[m[2]+2:m[3]]
            
            if i == 0:
                preYear = year
                
            if preYear != year:
                e4 = time.time()
                logging.debug( "\t"+str(tot)+" images took "+str(e4-s4)+" seconds for "+preYear+" year" )
                tot=0
                s4 = time.time()
                preYear = year

            stormNo = r.F1[m[4]+2:m[5]]
            f_ = r.F1[m[5]+2:m[6]]
            matFileName = r.F1[m[8]+2: len(r.F1)-4] + ".mat"

            #Reading File_Per_Year
            matFilesLocation = file_per_year[year][reg][stormNo][f_][0]
            matFiles = file_per_year[year][reg][stormNo][f_][1]

            try:
                ind = matFiles.index(matFileName)
            except:
                logging.debug("Error finding index of "+matFileName+" for ",year,stormNo,f_)
            matFile = matFiles[ind]

            #Reading .matFile
            lon,lat,diff = ReadMatFile("..\\"+matFilesLocation + matFile, freqArr)

            #Getting Lon and Lat corners
            loncorners,latcorners = GetLatLonCorners(r.CenLon, r.CenLat)

            #Creating Image
            fig, m = WorldMap(r.CenLon, r.CenLat, loncorners, latcorners)
            a,b = m(lon,lat)
            plt.pcolormesh(a,b,diff, cmap="jet_r")

            plt.tight_layout(pad=0)
            plt.savefig(r.FileName,bbox_inches = 'tight', pad_inches = 0)

            plt.close( fig )
            tot+=1
        
        e4 = time.time()
        logging.debug( "\t"+str(tot)+" images took "+str(e4-s4)+" seconds for "+preYear+" year" )
        
        e3 = time.time()
        logging.debug( "Time taken for "+reg+" region and "+freqArr[0]+" freq: "+str(e3-s3)+" seconds")
        logging.debug("===========================================")
        
    e2 = time.time()
    logging.debug( "Time taken for "+reg+" region: "+str(e2-s2)+" seconds")
    logging.debug("=========================================================================")
    
e1 = time.time()
logging.debug( "Total Time taken: "+str(e1-s1)+" seconds")
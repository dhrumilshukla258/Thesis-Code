import matplotlib.dates as mpldate
import pandas as pd
import numpy as np
import math
import json

def GetNormalDate( serial_date ):
    normal_date = np.empty( (serial_date.shape),dtype="datetime64[ns]" )
    for i in range(0,len(serial_date)):
        normal_date[i,0] = np.datetime64( mpldate.num2date(serial_date[i,0] ) )
    return normal_date

def GetSerialDateFromString( str_name ):
    return mpldate.datestr2num(str_name) 

def GetSerialDate( normal_date ):
    return mpldate.date2num(normal_date)

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
        print(error)

def GetJSONFiles(filename):
    jfile = open( filename )
    data = json.load( jfile )

    #Converting String Keys to int
    data = {int(k1): 
       { k2 : 
         { int(k3) : v3 for k3,v3 in v2.items() } 
         for k2,v2 in v1.items() 
        }  
        for k1,v1 in data.items() 
       }
    return data

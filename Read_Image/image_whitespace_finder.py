import sys
sys.path.append('..\\Helper_Class')

from helper_functions import GetJSONFiles
import cv2
import copy
import json
import time
import logging

data = json.load( open( "..//images_per_year6.json" ) )
images_per_year = copy.copy(data)

def ProbabilityOfWhiteSpace(img_cv):
    count = 0
    for i in range(img_cv.shape[0]):
        for j in range(img_cv.shape[1]):
            if ( img_cv[i][j][0] == 255 and img_cv[i][j][1] == 255 and img_cv[i][j][2] == 255):
                count+=1
    return count / (img_cv.shape[0]*img_cv.shape[1])


startMain = time.time()
mPath = "..//..//MyCreatedData//6//"
for region,v1 in data.items():
    path1 = mPath + region
    for storm_no,v2 in v1.items():
        print( region + " " + storm_no )
        path2 = path1 + "//" + storm_no
        for satellite,v3 in v2.items():
            path3 = path2 + "//"+ satellite
            for swath,v4 in v3.items():
                path4 = path3 + "//"+ swath
                for freq,v5 in v4.items():
                    path5 = path4 + "//"+ freq
                    images_per_year[region][storm_no][satellite][swath][freq] = []
                    for file in v5:
                        #start = time.time()
                        
                        img_path = path5 + "//"+ file
                        image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)  
                        image = cv2.resize(image, (360,360))
                        prob_of_color_image = 1 - ProbabilityOfWhiteSpace( image )
                        images_per_year[region][storm_no][satellite][swath][freq].append( [ file, prob_of_color_image ] )
                        
                        #end = time.time()
                        #print((end-start))

with open('..\\images_per_year6_with_prob.json', 'w') as outfile:
    json.dump( images_per_year, outfile, indent=4 )

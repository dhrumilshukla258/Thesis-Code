import sys

import copy
import json
import time
import logging


for year in range(5,20):
    data = json.load( open( "..//#NewImagePerYear//images_per_year" + str(year) +".json" ) )
    images_per_year = copy.copy(data)
    
    old_images_per_year = json.load( open( "..//#ImagePerYear//images_per_year" + str(year) +".json" ) )

    mPath = "..//..//MyCreatedData//"+ str(year) +"//"
    for region,v1 in data.items():
        for storm_no,v2 in v1.items():
            for satellite,v3 in v2.items():
                for swath,v4 in v3.items():
                    for freq,v5 in v4.items():
                        images_per_year[region][storm_no][satellite][swath][freq] = []
                        oldFiles = old_images_per_year[region][storm_no][satellite][swath][freq]
                        for file in v5:
                            #start = time.time()

                            for oF in oldFiles:
                                if oF[0] == file:
                                    images_per_year[region][storm_no][satellite][swath][freq].append( [file, oF[1]]  )
                                    break
                            
                            #end = time.time()
                            #print((end-start))

    with open('..\\images_per_year'+ str(year) + '_with_prob.json', 'w') as outfile:
        json.dump( images_per_year, outfile, indent=4 )

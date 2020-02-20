import os
import re
import json



for year in range(5,20): 
    images_per_year = {}
    for root, dirs, files in os.walk("..\\..\\MyCreatedData\\"+str(year), topdown=False):
        #print (dirs)
        #if "SSMIS\F" in root:
        #    print ( root )
        
        
        if len(files) == 0:
            continue
        
        if ".png" not in files[0]:
            continue
        
        m = [m.start() for m in re.finditer(r'\\', root)]
        
        print(root)
        region = root[ m[3]+1 : m[4] ]
        stormNo = int(root[ m[4]+1 : m[5] ])
        f_ =  root[ m[5]+1 : m[6] ]
        swath = root[ m[6]+1 : m[7] ]
        freq  = root[ m[7]+1 : ]
        
        if images_per_year.get(region) == None:
            images_per_year[region] = {}
        
        if images_per_year[region].get(stormNo) == None:
            images_per_year[region][stormNo] = {}
        
        if images_per_year[region][stormNo].get(f_) == None:
            images_per_year[region][stormNo][f_] = {}
        
        if images_per_year[region][stormNo][f_].get(swath) == None:
            images_per_year[region][stormNo][f_][swath] = {}
        
        images_per_year[region][stormNo][f_][swath][freq] = files
    
    fileName = '..\\images_per_year' + str(year) + '.json'
    with open(fileName, 'w') as outfile:
        json.dump( images_per_year, outfile, indent=2 )
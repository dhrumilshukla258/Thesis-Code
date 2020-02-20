import sys
sys.path.append('..\\Helper_Class')

from helper_functions import GetJSONFiles
import time
import logging
import json
logging.basicConfig(filename='..\\LogFile\\logFile_storm',level=logging.DEBUG)

data = GetJSONFiles("..\\files_per_year.json")
images_per_year = {}

from storm_reader import StormReader
'''
year = 6
region = 'WPAC'

for storm_no in range(26,27):
    print(storm_no)
    start = time.time()
    storm = data[year][region][storm_no]
    sR = StormReader( storm, year, region, storm_no )
    sR.ReadStorm()
    end = time.time()
    logging.debug( " " + str(year) + " " + str(region) + " " + str(storm_no) + " : " + str( (end-start)/60 ) + " minutes")

'''

startMain = time.time()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

logging.debug(" StartTime:- " + str(startMain) )

for year,v1 in data.items():
    if images_per_year.get(year) == None:
        images_per_year[ year ] = {}
        
    if year == 10 or year == 11 or year == 12 or year == 13 or year == 14 or year == 5  or year == 6:
        continue
    
    for region,v2 in v1.items():
        if images_per_year[year].get(region) == None:
            images_per_year[year][region] = {}
            
        for storm_no,v3 in v2.items():
            if images_per_year[year][region].get(storm_no) == None:
                images_per_year[year][region][storm_no] = {}
            
            storm = data[year][region][storm_no]
            start = time.time()
            
            sR = StormReader( storm, year, region, storm_no )
            sR.ReadStorm()
            images_per_year[year][region][storm_no] = sR.mImagesWithProbability
            
            end = time.time()
            logging.debug( " " + str(year) + " " + str(region) + " " + str(storm_no) + " : " + str( (end-start)/60 ) + " minutes")
    with open('..\\images_per_year'+str(year)+'.json','w') as outfile:
        json.dump(images_per_year[year], outfile,indent=2)
   

endMain = time.time()

with open('..\\images_per_year.json', 'w') as outfile:
    json.dump(images_per_year, outfile,indent=2)

logging.debug(" EndTime:- " + str(endMain) )
logging.debug( " Total time:- " + str( (endMain - startMain)/3600 ) + " hours" )

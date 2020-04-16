import os
import re
import json
import copy
files_per_year = {}
count=0

#Main Folder of the HardDrive
for root, dirs, files in os.walk("..\\CurrentData\\Passtimes_and_1C_Data", topdown=False):
    if "SSMIS\F" in root:
        m = [m.start() for m in re.finditer(r'\\', root)]
        year =  ( root[ m[3]+1:m[4] ] )
        region = root[ m[4]+1:m[5] ]
        stormNo =  ( root[ m[5]+1:m[6] ] )
        f_ =  root[ m[8]+1 : ]
        
        if files_per_year.get(year) == None:
            files_per_year[ year ] = {}
        if files_per_year[year].get(region) == None:
            files_per_year[year][region] = {}
        if files_per_year[year][region].get(stormNo) == None:
            files_per_year[year][region][stormNo] = {}
        
        files_per_year [year][region][stormNo][f_] = [ root + "\\" , files ] 

        count+=1
        #print(count)

data = copy.deepcopy(files_per_year)

#Reading BestTrack files
for k1,v1 in data.items():
    for k2,v2 in v1.items():
        for k3,v3 in v2.items():
            for k4,v4 in v3.items():
                st = "..\\CurrentData\\SHIPS_Navy_Combined\\raw_data\\" + k1 + "\\" + k2 + "\\" + k3
                try:
                    files = os.listdir(st)
                    if files_per_year[k1][k2].get(k3) != None:
                        files_per_year[k1][k2][k3]["BestTrack"] = [ st + "\\", files ]
                except:
                    if files_per_year[k1][k2].get(k3) != None:
                        del files_per_year[k1][k2][ k3 ]

#Creating JSON file
with open('files_per_year.json', 'w') as outfile:
    json.dump(files_per_year, outfile,indent=2)
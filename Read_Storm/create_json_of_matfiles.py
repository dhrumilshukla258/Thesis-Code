import os
import re
import json
import copy

# Locating SSMIS Mat files
matfiles_and_best_track = {}
count=0

# Main Folder of the HardDrive
for root, dirs, files in os.walk("..\\..\\CurrentData\\Passtimes_and_1C_Data\\raw_data", topdown=False):
    if "SSMIS\F" in root:
        m = [m.start() for m in re.finditer(r'\\', root)]
        year =  ( root[ m[4]+1:m[5] ] )
        region = root[ m[5]+1:m[6] ]
        stormNo =  ( root[ m[6]+1:m[7] ] )
        f_ =  root[ m[9]+1 : ]

        if matfiles_and_best_track.get(region) == None:
            matfiles_and_best_track[ region ] = {}
        if matfiles_and_best_track[region].get(year) == None:
            matfiles_and_best_track[region][year] = {}
        if matfiles_and_best_track[region][year].get(stormNo) == None:
            matfiles_and_best_track[region][year][stormNo] = {}
        
        matfiles_and_best_track[region][year][stormNo][f_] = [ root + "\\" , files ] 

        count+=1

print("Total SSMIS Folders found: "+str(count))

# Locating BestTrack files
data = copy.deepcopy(matfiles_and_best_track)

for region,v1 in data.items():
    for year,v2 in v1.items():
        for stormNo,v3 in v2.items():
            for f_,v4 in v3.items():
                st = "..\\..\\CurrentData\\SHIPS_Navy_Combined\\raw_data\\" + year + "\\" + region + "\\" + stormNo
                try:
                    files = os.listdir(st)
                    if matfiles_and_best_track[region][year].get(stormNo) != None:
                        matfiles_and_best_track[region][year][stormNo]["BestTrack"] = [ st + "\\", files ]
                except:
                    if matfiles_and_best_track[region][year].get(stormNo) != None:
                        del matfiles_and_best_track[region][year][ stormNo ]

#Creating JSON file
with open('matfiles_and_best_track.json', 'w') as outfile:
    json.dump(matfiles_and_best_track, outfile,indent=2)
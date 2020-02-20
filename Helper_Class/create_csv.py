import pandas as pd
from helper_function import GetJSONFiles

imgFreq = {}
fList = ['19H','19V','22V','37V','37H','91H','91V','150H','183_1H','183_3H','183_7H']
rList = ['ATL','CPAC','EPAC','IO','SHEM','WPAC']
for r in rList:
    imgFreq[r] = {}
    for f in fList:
        imgFreq[r][f] = []


for year in range(5,20):
    data = GetJSONFiles("..//ImagesPerYear//images_per_year_"+ str(year) + ".json")
    for region,v1 in data.items():
        for storm_no,v2 in v1.items():
            st_data = pd.read_csv("..//..//MyCreatedData//"+str(year)+"//"+region+"//"+str(storm_no)+"//StormData.txt",sep="\t")
            st_data = st_data.drop_duplicates()
            for satellite,v3 in v2.items():
                for swath,v4 in v3.items():
                    for freq,v5 in v4.items():
                        for file in v5:
                            imgPath = "..//..//MyCreatedData//"+ str(year) +"//"+region+"//"+str(storm_no)+"//"+satellite+"//"+swath+"//"+freq+"//"+file[0]
                            t = st_data.loc[st_data['FileName'] == file[0]]
                            if len(t) == 1:
                                cenLon = t.iloc[0,1]
                                cenLat = t.iloc[0,2]
                                pressure = t.iloc[0,3]
                                wind = t.iloc[0,4]
                                imgFreq[region][freq].append([ imgPath,cenLon,cenLat,pressure,wind,file[1] ]) 
                            else:
                                print("Error No Data for this image")   

for r in rList:
    for f in fList:
        t = pd.DataFrame(imgFreq[r][f])
        t.columns = ["FileName", "CenLon", "CenLat", "Pressure","Wind","Area"]
        t.to_csv("..//ImagesPerFreq//"+r+"_"+f+'.csv')
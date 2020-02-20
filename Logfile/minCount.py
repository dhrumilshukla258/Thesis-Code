f = open("logFile_storm","r")
totalMin = 0
for x in f:
    word = x.split(" ")
    if len(word) > 4:
        totalMin += float(word[5])
print(totalMin/60)

from itertools import izip

fileA = open("edge.csv")
fileB = open("server.csv")

for lineA, lineB in izip(fileA, fileB):
    edgeTime = lineA.split(",")[4]
    edgeEnergy = lineA.split(",")[5]
    print "%s,%s,%s" % (lineB.rstrip(), edgeTime, edgeEnergy.rstrip())


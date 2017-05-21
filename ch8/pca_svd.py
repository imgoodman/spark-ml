from pyspark import SparkContext

sc=SparkContext("local[2]","spark pca and svd app")

file_path="/usr/bigdata/data/lfw/*"

rdd=sc.wholeTextFiles(file_path)

#print rdd.first()

files=rdd.map(lambda (fileName, fileContent) : fileName.replace("file:",""))

print files.first()
print files.count()

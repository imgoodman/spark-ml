from pyspark import SparkContext

sc=SparkContext("local[2]","spark tf-idf app")

file_path="/usr/bigdata/data/20news-bydate-train/*"

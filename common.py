import os

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

script_dir = os.path.dirname(__file__)
data_dir = "csv/"
parquet_dir = "parquets/"
model_dir = "models/"
executor_memory = "32G"
driver_memory = "32G"
max_result_size = "0"

def initSpark():
    conf = SparkConf().setAppName("CS145 Project")
    conf = conf.setMaster("local[*]").set(
        "spark.executor.memory",
        executor_memory).set(
        "spark.driver.memory",
        driver_memory).set(
        "spark.driver.maxResultSize",
        max_result_size)
    sc = SparkContext(conf=conf)
    return SQLContext(sc)

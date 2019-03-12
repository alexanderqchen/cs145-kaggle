import os

from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import SQLContext
from pyspark.sql.functions import avg, col, udf
from pyspark.sql.types import FloatType

script_dir = os.path.dirname(__file__)
space_dir = script_dir
data_dir = "csv/"
parquet_dir = "parquets/"
models_dir = "models/"


def get_initial_dfs(context, saved_initial_dfs):
    if not saved_initial_dfs:
        genome_scores = context.read.format("csv").option("header", "true").load(
            os.path.join(script_dir, data_dir + "genome-scores.csv"))
        genome_scores = genome_scores.select(
            col("movieId").cast("int").alias("movieId"),
            col("tagId").cast("int").alias("tagId"),
            col("relevance").cast("float").alias("relevance"))
        train_ratings = context.read.format("csv").option("header", "true").load(
            os.path.join(script_dir, data_dir + "train_ratings.csv"))
        train_ratings = train_ratings.select(
            col("userId").cast("int").alias("userId"),
            col("movieId").cast("int").alias("movieId"),
            col("rating").cast("float").alias("rating"))
        val_ratings = context.read.format("csv").option("header", "true").load(
            os.path.join(script_dir, data_dir + "val_ratings.csv"))
        val_ratings = val_ratings.select(
            col("userId").cast("int").alias("userId"),
            col("movieId").cast("int").alias("movieId"),
            col("rating").cast("float").alias("rating"))
        test_ratings = context.read.format("csv").option("header", "true").load(
            os.path.join(script_dir, data_dir + "test_ratings.csv"))
        test_ratings = test_ratings.select(
            col("Id").cast("int").alias("Id"),
            col("userId").cast("int").alias("userId"),
            col("movieId").cast("int").alias("movieId"))
        print("Loaded genome scores and ratings.")

        print("Calculating movie vector.")
        movie_vectors = genome_scores.groupBy("movieId").pivot("tagId").sum("relevance")

        print("Calculating user vector.")
        user_vectors = train_ratings.join(movie_vectors, train_ratings.movieId == movie_vectors.movieId).select(
            "userId",
            "rating",
            *((col(c) * (train_ratings.rating - 3) / 2).alias(c) for c in movie_vectors.columns[1:])
        ).groupBy("userId").agg(*(avg(c).alias(c) for c in movie_vectors.columns[1:]))

        print("Saving parquets.")
        movie_vectors.write.parquet(os.path.join(space_dir, parquet_dir + "movie_vectors.parquet"))
        user_vectors.write.parquet(os.path.join(space_dir, parquet_dir + "user_vectors.parquet"))
        train_ratings.write.parquet(os.path.join(space_dir, parquet_dir + "train_ratings.parquet"))
        val_ratings.write.parquet(os.path.join(space_dir, parquet_dir + "val_ratings.parquet"))
        test_ratings.write.parquet(os.path.join(space_dir, parquet_dir + "test_ratings.parquet"))
    else:
        movie_vectors = context.read.parquet(os.path.join(space_dir, parquet_dir + "movie_vectors.parquet"))
        user_vectors = context.read.parquet(os.path.join(space_dir, parquet_dir + "user_vectors.parquet"))
        train_ratings = context.read.parquet(os.path.join(space_dir, parquet_dir + "train_ratings.parquet"))
        val_ratings = context.read.parquet(os.path.join(space_dir, parquet_dir + "val_ratings.parquet"))
        test_ratings = context.read.parquet(os.path.join(space_dir, parquet_dir + "test_ratings.parquet"))

    return movie_vectors, user_vectors, train_ratings, val_ratings, test_ratings


def calculate_paired_vectors(context, movie, user, train, val, test, saved_paired_vectors):
    if not saved_paired_vectors:
        movie_alias = movie.select("movieId", *((col(c)).alias("movie_" + c) for c in movie.columns[1:]))
        user_alias = user.select("userId", *((col(c)).alias("user_" + c) for c in user.columns[1:]))

        print("Calculating training vector.")
        train_vec = train.join(movie_alias, "movieId").join(user_alias, "userId").select(
            "rating",
            *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))

        print("Calculating validation vector.")
        val_vec = val.join(movie_alias, "movieId").join(user_alias, "userId").select(
            "rating",
            *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))

        print("Calculating test vector.")
        test_vec = test.join(movie_alias, "movieId").join(user_alias, "userId").select(
            "Id",
            *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))

        print("Saving parquets.")
        train_vec.write.parquet(os.path.join(space_dir, parquet_dir + "train_vec.parquet"))
        val_vec.write.parquet(os.path.join(space_dir, parquet_dir + "val_vec.parquet"))
        test_vec.write.parquet(os.path.join(space_dir, parquet_dir + "test_vec.parquet"))
    else:
        train_vec = context.read.parquet(os.path.join(space_dir, parquet_dir + "train_vec.parquet"))
        val_vec = context.read.parquet(os.path.join(space_dir, parquet_dir + "val_vec.parquet"))
        test_vec = context.read.parquet(os.path.join(space_dir, parquet_dir + "test_vec.parquet"))

    return train_vec, val_vec, test_vec


def train_model(saved_trained_model, train):
    if not saved_trained_model:
        assembler = VectorAssembler(
            inputCols=[c for c in train.columns[1:]],
            outputCol='features')

        lr = LinearRegression(labelCol="rating", featuresCol="features")
        print("Training model...")
        model = lr.fit(assembler.transform(train))
        model.save(os.path.join(space_dir, models_dir + "movies.model"))
    else:
        model = LinearRegressionModel.load(os.path.join(space_dir, models_dir + "movies.model"))
    return model


def print_val_score(model, val):
    assembler = VectorAssembler(
        inputCols=[c for c in val.columns[1:]],
        outputCol='features')

    print("Getting validation score...")
    print(RegressionEvaluator(labelCol="rating").evaluate(model.transform(assembler.transform(val))))


def cutoff(x):
    if x > 5.0:
        return 5.0
    elif x < 0.5:
        return 0.5
    return x


def predict_scores(model, test):
    assembler = VectorAssembler(
        inputCols=[c for c in test.columns[1:]],
        outputCol="features")
    cutoff_udf = udf(cutoff, FloatType())
    result = model.transform(assembler.transform(test)).select(
        "Id",
        cutoff_udf("prediction").alias("rating"))

    print("Generating result.csv")
    result.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(
        os.path.join(script_dir, data_dir + "result.csv"))


def main(context):
    saved_initial_dfs = True
    saved_paired_vectors = True
    saved_trained_model = True

    movie_vectors, user_vectors, train_ratings, val_ratings, test_ratings = get_initial_dfs(context, saved_initial_dfs)
    train_vec, val_vec, test_vec = calculate_paired_vectors(context, movie_vectors, user_vectors, train_ratings,
                                                            val_ratings, test_ratings, saved_paired_vectors)
    model = train_model(saved_trained_model, train_vec)
    #print_val_score(model, val_vec)
    predict_scores(model, test_vec)


if __name__ == "__main__":
    conf = SparkConf().setAppName("CS145 Project")
    conf = conf.setMaster("local[*]").set("spark.executor.memory", "768G").set("spark.driver.memory", "768G").set(
        "spark.local.dir", "/tmp")
    sc = SparkContext(conf=conf)
    sql_context = SQLContext(sc)
    main(sql_context)

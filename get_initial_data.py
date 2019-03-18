import os

from common import script_dir, data_dir, parquet_dir, initSpark

from pyspark.sql.functions import avg, col

def main():
    context = initSpark()
    print("Loading data...")
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
    val_ratings = train_ratings.select(
        col("userId").cast("int").alias("userId"),
        col("movieId").cast("int").alias("movieId"),
        col("rating").cast("float").alias("rating"))
    test_ratings = context.read.format("csv").option("header", "true").load(
        os.path.join(script_dir, data_dir + "test_ratings.csv"))
    test_ratings = test_ratings.select(
        col("Id").cast("int").alias("Id"),
        col("userId").cast("int").alias("userId"),
        col("movieId").cast("int").alias("movieId"))

    print("Calculating movie vector...")
    movie_vectors = genome_scores.groupBy("movieId").pivot("tagId").sum("relevance")

    print("Calculating user vector...")
    user_vectors = train_ratings.join(movie_vectors, train_ratings.movieId == movie_vectors.movieId).select(
        "userId",
        "rating",
        *((col(c) * (train_ratings.rating - 3.535) / 1.05).alias(c) for c in movie_vectors.columns[1:])
    ).groupBy("userId").agg(*(avg(c).alias(c) for c in movie_vectors.columns[1:]))

    print("Saving movie vector...")
    movie_vectors.write.parquet(os.path.join(script_dir, parquet_dir + "movie_vectors.parquet"))
    print("Saving user vector...")
    user_vectors.write.parquet(os.path.join(script_dir, parquet_dir + "user_vectors.parquet"))
    print("Saving training ratings...")
    train_ratings.write.parquet(os.path.join(script_dir, parquet_dir + "train_ratings.parquet"))
    print("Saving validation ratings...")
    val_ratings.write.parquet(os.path.join(script_dir, parquet_dir + "val_ratings.parquet"))
    print("Saving test ratings...")
    test_ratings.write.parquet(os.path.join(script_dir, parquet_dir + "test_ratings.parquet"))
    print("Done.")


if __name__ == "__main__":
    main()

import os

from pyspark.sql.functions import col

from common import script_dir, parquet_dir, init_spark


def main():
    context = init_spark()
    print("Loading data...")
    movie = context.read.parquet(os.path.join(script_dir, parquet_dir + "movie_vectors.parquet"))
    user = context.read.parquet(os.path.join(script_dir, parquet_dir + "user_vectors.parquet"))
    train = context.read.parquet(os.path.join(script_dir, parquet_dir + "train_ratings.parquet"))
    val = context.read.parquet(os.path.join(script_dir, parquet_dir + "val_ratings.parquet"))
    test = context.read.parquet(os.path.join(script_dir, parquet_dir + "test_ratings.parquet"))

    movie_alias = movie.select("movieId", *((col(c)).alias("movie_" + c) for c in movie.columns[1:]))
    user_alias = user.select("userId", *((col(c)).alias("user_" + c) for c in user.columns[1:]))

    print("Calculating training vector....")
    train_vec = train.join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "rating",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))
    print("Calculating validation vector...")
    val_vec = val.join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "rating",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))
    print("Calculating test vector....")
    test_vec = test.join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "Id",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:])).orderBy("Id")

    print("Saving training vector...")
    train_vec.write.parquet(os.path.join(script_dir, parquet_dir + "train_vec_spark.parquet"))
    print("Saving validation vector...")
    val_vec.write.parquet(os.path.join(script_dir, parquet_dir + "val_vec_spark.parquet"))
    print("Saving test vector...")
    test_vec.write.parquet(os.path.join(script_dir, parquet_dir + "test_vec_spark.parquet"))
    print("Done.")


if __name__ == "__main__":
    main()

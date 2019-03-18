import os

from common import script_dir, parquet_dir, initSpark

from pyspark.sql.functions import col

def main():
    context = initSpark()
    print("Loading data...")
    movie = context.read.parquet(os.path.join(script_dir, parquet_dir + "movie_vectors.parquet"))
    user = context.read.parquet(os.path.join(script_dir, parquet_dir + "user_vectors.parquet"))
    train = context.read.parquet(os.path.join(script_dir, parquet_dir + "train_ratings.parquet"))
    val = context.read.parquet(os.path.join(script_dir, parquet_dir + "val_ratings.parquet"))
    test = context.read.parquet(os.path.join(script_dir, parquet_dir + "test_ratings.parquet"))

    movie_alias = movie.select("movieId", *((col(c)).alias("movie_" + c) for c in movie.columns[1:]))
    user_alias = user.select("userId", *((col(c)).alias("user_" + c) for c in user.columns[1:]))

    print("Calculating training vector....")
    train_vec = train.sample(False, 0.01).join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "rating",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))
    print("Calculating validation vector...")
    val_vec = val.sample(False, 0.0001).join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "rating",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))
    print("Calculating test vector....")
    test_vec = test.join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "Id",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))

    print("Saving train vector...")
    train_vec.write.parquet(os.path.join(script_dir, parquet_dir + "train_vec.parquet"))
    print("Saving validation vector...")
    val_vec.write.parquet(os.path.join(script_dir, parquet_dir + "val_vec.parquet"))
    print("Saving test vector...")
    test_vec.write.parquet(os.path.join(script_dir, parquet_dir + "test_vec.parquet"))
    print("Done.")


if __name__ == "__main__":
    main()

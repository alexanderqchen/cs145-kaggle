import os

import pandas as pd
from sklearn.decomposition import IncrementalPCA

from common import script_dir, parquet_dir


def main():
    print("Loading training data...")
    train_vec = pd.read_parquet(os.path.join(script_dir, parquet_dir + "train_vec_spark.parquet")).fillna(0)
    train_x = train_vec.drop("rating", axis=1)
    train_y = train_vec["rating"]
    print("Loading validation data...")
    val_vec = pd.read_parquet(os.path.join(script_dir, parquet_dir + "val_vec_spark.parquet")).fillna(0)
    val_x = val_vec.drop("rating", axis=1)
    val_y = val_vec["rating"]
    print("Loading test data...")
    test_vec = pd.read_parquet(os.path.join(script_dir, parquet_dir + "test_vec_spark.parquet")).fillna(0)
    test_x = test_vec.drop("Id", axis=1)
    test_id = test_vec["Id"]

    pca = IncrementalPCA(n_components=20)
    print("Fitting PCA model...")
    model = pca.fit(train_x)
    print("Reducing training dimensionality...")
    train_vec = pd.DataFrame(model.transform(train_x)).astype("float32")
    train_vec.insert(0, "rating", train_y)
    train_vec.columns = train_vec.columns.astype(str)
    print("Reducing validation dimensionality...")
    val_vec = pd.DataFrame(model.transform(val_x)).astype("float32")
    val_vec.insert(0, "rating", val_y)
    val_vec.columns = val_vec.columns.astype(str)
    print("Reducing test dimensionality...")
    test_vec = pd.DataFrame(model.transform(test_x)).astype("float32")
    test_vec.insert(0, "Id", test_id)
    test_vec.columns = test_vec.columns.astype(str)

    print("Saving training vector...")
    train_vec.to_parquet(os.path.join(script_dir, parquet_dir + "train_vec.parquet"))
    print("Saving validation vector...")
    val_vec.to_parquet(os.path.join(script_dir, parquet_dir + "val_vec.parquet"))
    print("Saving test vector...")
    test_vec.to_parquet(os.path.join(script_dir, parquet_dir + "test_vec.parquet"))
    print("Done.")


if __name__ == "__main__":
    main()

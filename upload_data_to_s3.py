import io
import boto3
import os
import pandas as pd

from common import script_dir, parquet_dir

from sagemaker.amazon.common import write_numpy_to_dense_tensor

def main():
    print("Loading data...")
    train = pd.read_parquet(os.path.join(script_dir, parquet_dir + "train_vec.parquet")).fillna(0).to_numpy().astype("float32")
    test = pd.read_parquet(os.path.join(script_dir, parquet_dir + "test_vec.parquet")).fillna(0).to_numpy().astype("float32")

    bucket = "cs145sagemaker"
    train_key = "trainingData"
    train_location = "s3://{}/{}".format(bucket, train_key)
    print("Training data will be uploaded to: {}".format(train_location))
    test_key = "testData"
    test_location = "s3://{}/{}".format(bucket, test_key)
    print("Test data will be uploaded to: {}".format(train_location))

    print("Converting training data...")
    buf = io.BytesIO()
    write_numpy_to_dense_tensor(buf, train[:,1:], train[:,0])
    buf.seek(0)

    print("Uploading training data...")
    boto3.resource("s3").Bucket(bucket).Object(train_key).upload_fileobj(buf)

    print("Converting test data...")
    buf = io.BytesIO()
    write_numpy_to_dense_tensor(buf, test[:,1:], test[:,0])
    buf.seek(0)

    print("Uploading test data...")
    boto3.resource("s3").Bucket(bucket).Object(test_key).upload_fileobj(buf)
    print("Done.")


if __name__ == "__main__":
    main()

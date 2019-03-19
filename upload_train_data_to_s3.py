import io
import boto3
import os
import pandas as pd

from common import script_dir, parquet_dir

from sagemaker.amazon.common import write_numpy_to_dense_tensor

def main():
    print("Loading data...")
    train = pd.read_parquet(os.path.join(script_dir, parquet_dir + "train_vec.parquet")).to_numpy()

    bucket = "cs145projectucla"
    data_key = "trainingData"
    data_location = "s3://{}/{}".format(bucket, data_key)
    print("Training data will be uploaded to: {}".format(data_location))

    print("Converting data...")
    buf = io.BytesIO()
    write_numpy_to_dense_tensor(buf, train[:,1:], train[:,0])
    buf.seek(0)

    print("Uploading data...")
    boto3.resource("s3").Bucket(bucket).Object(data_key).upload_fileobj(buf)
    print("Done.")


if __name__ == "__main__":
    main()

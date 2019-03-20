import math
import os

import pandas as pd
from joblib import dump
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from common import script_dir, parquet_dir, model_dir


def main():
    print("Loading data...")
    train = pd.read_parquet(os.path.join(script_dir, parquet_dir + "train_vec.parquet"))
    val = pd.read_parquet(os.path.join(script_dir, parquet_dir + "val_vec.parquet"))
    train_x = train.drop("rating", axis=1).fillna(0)
    train_y = train["rating"].fillna(0)
    val_x = val.drop("rating", axis=1).fillna(0)
    val_y = val["rating"].fillna(0)

    print("Training model...")
    nn = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, verbose=True)
    nn.fit(train_x, train_y)
    print("Validation RMSE: ", math.sqrt(mean_squared_error(val_y, nn.predict(val_x))))
    print("Saving model...")
    dump(nn, os.path.join(script_dir, model_dir + "nn.model"))
    print("Done.")


if __name__ == "__main__":
    main()

# COM SCI 145 Kaggle Contest - Team Combination Pizza Hut and Taco Bell
# Jennie Zheng, Yijing Zhou, Alex Chen, Michael Wu, Danny Nguyen

## Dependencies
The program assumes that python's surprise library is installed. If not, please run
`pip3 install surprise`

### csv
All the relevant kaggle-competition data should be downloaded and stored here.

### Running Program
To run the program, issue `python svd.pp` to begin training the ensemble of 50 SVD models on the raw data.
This may take awhile as each of the models is trained for 35 epochs.

### Notes
The python surprise package works directly with the raw {userId, movieId, rating} tuples. There was no need
to preprocess the data or to incorporate any additional features into training our models.
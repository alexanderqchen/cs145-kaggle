# COM SCI 145 Kaggle Contest

## Organization
### csv
The config file assumes you store all of your csv files in a folder named csv.
### data
The data folder contains classes or modules relevant to data and data processing.
### ml
The ml folder contains classes that define classifiers.

## Conventions
### Internal Data Structures
Internally, all data is passed around as lists of dictionaries. Each dictionary maps the column name to the value for that instance. E.g. the csv file
```
userId,movieId,rating
1,1,4.0
1,2,3.5
```
would be structured like
```
[
  {
    'userId': 1,
    'movieId': 1,
    'rating': 4.0
  },
  {
    'userId': 1,
    'movieId': 2,
    'rating': 3.5
  }
]
```

## Note
We gon win this thing ez.

# COM SCI 145 Kaggle Contest

## Organization
### csv
The config file assumes you store all of your csv files in a folder named csv.
### src
The source folder contains following sub-folders: **data**, **ml**, **visual**.
### data
The data sub-folder contains classes or modules relevant to data and data processing.
### ml
The ml sub-folder contains classes that define classifiers.
### visual 
The visual sub-folder contains Jupyter notebook that provides data visualization.

## Conventions
### Internal Data Structures
Internally, all data is passed around as numpy arrays or pandas dataframe . Each numpy array contains a header that will allow fast access to the value for that instance. E.g. the csv file
```
userId,movieId,rating
1,1,4.0
1,2,3.5
```
would be structured like
```
[[1.  1.  4. ]
 [1.  2.  3.5]]
```
## Note
We gon win this thing ez. 
I concur. 

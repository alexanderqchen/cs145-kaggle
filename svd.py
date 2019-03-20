from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV

# Load the movielens-100k dataset (download it if needed).
print("----Loading training----")
reader = Reader(line_format='user item rating', sep=',')
train_data = Dataset.load_from_file('./csv/train_ratings.csv', reader=reader)
trainset = train_data.build_full_trainset()

print("----Loading testing----")
test_reader = Reader(line_format='rating user item', sep=',')
test_data = Dataset.load_from_file('./csv/test_ratings.csv', reader=test_reader)
testset = test_data.build_full_trainset()
testset = testset.build_testset()

# Use the famous SVD algorithm.
NUM_MODELS = 50
NUM_EPOCHS = 35
algo = []
predict = []

print("----Training Model----")
for i in range(NUM_MODELS):
    print("Training svd Model: " + str(i))

    algo.append(SVD(verbose=True, n_epochs = NUM_EPOCHS))
    predict.append(algo[i].fit(trainset).test(testset))

print("----Predicting----")
f = open('./SVD_ensemble_results_1.csv', 'w+')
f.write('Id,rating\n')

for i in range(len(predict[0])):
    pred_rating = 0.0
    for j in range(NUM_MODELS):
        pred_rating += predict[j][i][3]
    pred_rating /= float(NUM_MODELS)

    f.write(str(int(predict[0][i][2])) +"," + str(pred_rating) + "\n")
    if (i % 10000 == 0):
        f.flush()
        print("movie: " + str(i))


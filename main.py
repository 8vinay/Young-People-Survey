import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

print("Reading data...\n")
df = pd.read_csv("responses.csv")
columns = pd.read_csv("columns.csv")

print("Preprocessing data...\n")
music = df.iloc[:,0:19]
movies = df.iloc[:,19:31]
hobbies = df.iloc[:,31:63]
phobias = df.iloc[:,63:73]
health = df.iloc[:,73:76]
personality = df.iloc[:, 76:133]
spending = df.iloc[:,133:140]
demographics = df.iloc[:,140:150]

music = music.replace("nan", np.nan)
music = music.replace("NaN", np.nan)

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp.fit(music)
music_data = imp.transform(music)
music = pd.DataFrame(data=music_data[:,:],
                     index=[i for i in range(len(music_data))],
                     columns=music.columns.tolist())

movies = movies.replace("nan", np.nan)
movies = movies.replace("NaN", np.nan)

imp.fit(movies)
movies_data = imp.transform(movies)
movies = pd.DataFrame(data=movies_data[:,:],
                     index=[i for i in range(len(movies_data))],
                     columns=movies.columns.tolist())

hobbies = hobbies.replace("nan", np.nan)
hobbies = hobbies.replace("NaN", np.nan)

imp.fit(hobbies)
hobbies_data = imp.transform(hobbies)
hobbies = pd.DataFrame(data=hobbies_data[:,:],
                     index=[i for i in range(len(hobbies_data))],
                     columns=hobbies.columns.tolist())

phobias = phobias.replace("nan", np.nan)
phobias = phobias.replace("NaN", np.nan)

imp.fit(phobias)
phobias_data = imp.transform(phobias)
phobias = pd.DataFrame(data=phobias_data[:,:],
                     index=[i for i in range(len(phobias_data))],
                     columns=phobias.columns.tolist())

health["Smoking"].unique()

for i in health["Smoking"]:
    if i == "never smoked":
        health.replace(i, 1.0, inplace=True)
    elif i == "tried smoking":
        health.replace(i, 2.0, inplace=True)
    elif i == "former smoker":
        health.replace(i, 3.0, inplace=True)
    elif i == "current smoker":
        health.replace(i, 4.0, inplace=True)

health["Alcohol"].unique()
for i in health["Alcohol"]:
    if i == "never":
        health.replace(i, 1.0, inplace=True)
    elif i == "social drinker":
        health.replace(i, 2.0, inplace=True)
    elif i == "drink a lot":
        health.replace(i, 3.0, inplace=True)

health = health.replace("nan", np.nan)
health = health.replace("NaN", np.nan)

imp.fit(health)
health_data = imp.transform(health)
health = pd.DataFrame(data=health_data[:,:],
                     index=[i for i in range(len(health_data))],
                     columns=health.columns.tolist())

personality["Punctuality"].unique()
for i in personality["Punctuality"]:
    if i == "i am often running late":
        personality.replace(i, 1.0, inplace=True)
    elif i == "i am always on time":
        personality.replace(i, 2.0, inplace=True)
    elif i == "i am often early":
        personality.replace(i, 3.0, inplace=True)

personality["Lying"].unique()        
for i in personality["Lying"]:
    if i == "never":
        personality.replace(i, 1.0, inplace=True)
    elif i == "only to avoid hurting someone":
        personality.replace(i, 2.0, inplace=True)
    elif i == "sometimes":
        personality.replace(i, 3.0, inplace=True)
    elif i == "everytime it suits me":
        personality.replace(i, 4.0, inplace=True)

personality["Internet usage"].unique()
for i in personality["Internet usage"]:
    if i == "no time at all":
        personality.replace(i, 1.0, inplace=True)
    elif i == "less than an hour a day":
        personality.replace(i, 2.0, inplace=True)
    elif i == "few hours a day":
        personality.replace(i, 3.0, inplace=True)
    elif i == "most of the day":
        personality.replace(i, 4.0, inplace=True)

personality = personality.replace("nan", np.nan)
personality = personality.replace("NaN", np.nan)

imp.fit(personality)
personality_data = imp.transform(personality)
personality = pd.DataFrame(data=personality_data[:,:],
                     index=[i for i in range(len(personality_data))],
                     columns=personality.columns.tolist())

spending = spending.replace("nan", np.nan)
spending = spending.replace("NaN", np.nan)

imp.fit(spending)
spending_data = imp.transform(spending)
spending = pd.DataFrame(data=spending_data[:,:],
                     index=[i for i in range(len(spending_data))],
                     columns=spending.columns.tolist())

for i in demographics["Gender"]:
    if i == "female":
        demographics.replace(i, 1.0, inplace=True)
    elif i == "male":
        demographics.replace(i, 2.0, inplace=True)

for i in demographics["Left - right handed"]:
    if i == "right handed":
        demographics.replace(i, 1.0, inplace=True)
    elif i == "left handed":
        demographics.replace(i, 2.0, inplace=True)

demographics["Education"].unique()
for i in demographics["Education"]:
    if i == "currently a primary school pupil":
        demographics.replace(i, 1.0, inplace=True)
    elif i == "primary school":
        demographics.replace(i, 2.0, inplace=True)
    elif i == "secondary school":
        demographics.replace(i, 3.0, inplace=True)
    elif i == "college/bachelor degree":
        demographics.replace(i, 4.0, inplace=True)
    elif i == "masters degree":
        demographics.replace(i, 5.0, inplace=True)
    elif i == "doctorate degree":
        demographics.replace(i, 6.0, inplace=True)

demographics["Only child"].unique()
for i in demographics["Only child"]:
    if i == "yes":
        demographics.replace(i, 1.0, inplace=True)
    elif i == "no":
        demographics.replace(i, 2.0, inplace=True)
        
demographics["Village - town"].unique()
for i in demographics["Village - town"]:
    if i=="village":
        demographics.replace(i, 1.0, inplace=True)
    elif i=="city":
        demographics.replace(i, 2.0, inplace=True)

demographics["House - block of flats"].unique()
for i in demographics["House - block of flats"]:
    if i == "block of flats":
        demographics.replace(i, 1.0, inplace=True)
    elif i == "house/bungalow":
        demographics.replace(i, 2.0, inplace=True)

for i in demographics["Age"]:
    if (15 <= i < 19):
        demographics["Age"].replace(i, 1.0, inplace=True)
    elif (19 <= i < 23):
        demographics["Age"].replace(i, 2.0, inplace=True)
    elif (23 <= i < 27):
        demographics["Age"].replace(i, 3.0, inplace=True)
    elif (27 <= i < 31):
        demographics["Age"].replace(i, 4.0, inplace=True)
        
for i in demographics["Height"]:
    if (i >= 180):
        demographics["Height"].replace(i, 1.0, inplace=True)
    elif (170 <= i < 180):
        demographics["Height"].replace(i, 2.0, inplace=True)
    elif (160 <= i < 170):
        demographics["Height"].replace(i, 3.0, inplace=True)
    elif (i < 160):
        demographics["Height"].replace(i, 4.0, inplace=True)

for i in demographics["Weight"]:
    if (i >= 100):
        demographics["Weight"].replace(i, 1.0, inplace=True)
    elif (80 <= i < 100):
        demographics["Weight"].replace(i, 2.0, inplace=True)
    elif (60 <= i < 80):
        demographics["Weight"].replace(i, 3.0, inplace=True)
    elif (i < 60):
        demographics["Weight"].replace(i, 4.0, inplace=True)

demographics["Number of siblings"].unique()
for i in demographics["Number of siblings"]:
    if i == 0:
        demographics["Number of siblings"].replace(i, 0.0, inplace=True)
    elif i == 1:
        demographics["Number of siblings"].replace(i, 1.0, inplace=True)
    elif i == 2:
        demographics["Number of siblings"].replace(i, 2.0, inplace=True)
    elif i == 3:
        demographics["Number of siblings"].replace(i, 3.0, inplace=True)
    elif i == 4:
        demographics["Number of siblings"].replace(i, 4.0, inplace=True)
    elif i > 4:
        demographics["Number of siblings"].replace(i, 5.0, inplace=True)

demographics = demographics.replace("nan", np.nan)
demographics = demographics.replace("NaN", np.nan)

imp.fit(demographics)
demographics_data = imp.transform(demographics)
demographics = pd.DataFrame(data=demographics_data[:,:],
                     index=[i for i in range(len(demographics_data))],
                     columns=demographics.columns.tolist())
        
data_all = music.join(movies.join(hobbies.join(phobias.join(health.join(
            personality.join(spending.join(demographics)))))))

from sklearn import model_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

print("Selecting the best 100 features from the data...\n")
y_all = [i for i in data_all.iloc[:, 95]]
responses_train = data_all.drop(["Empathy"], axis=1)
x_all = responses_train.iloc[:, :].values

test = SelectKBest(score_func=chi2, k=100)
bestfit = test.fit(x_all, y_all)
features = bestfit.transform(x_all)
#print(features.shape)
x_all=features

y = y_all[:int(len(y_all)*0.8)]
y_test = y_all[int(len(y_all)*0.8):]
x = x_all[:int(len(x_all)*0.8)]
x_test = x_all[int(len(x_all)*0.8):]

x_train, x_dev, y_train, y_dev = model_selection.train_test_split(x, y, test_size=20, random_state=50)

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print("Evaluating few classifiers on the training data using cross validation...\n")
kfold = model_selection.KFold(n_splits=10, random_state=50)
pred = model_selection.cross_val_score(DummyClassifier(strategy='most_frequent'), x_train, y_train, cv=kfold, scoring='accuracy')
print("accuracy for most frequent classifier: " , pred.mean()*100, "%")

kfold = model_selection.KFold(n_splits=10, random_state=50)
pred = model_selection.cross_val_score(DummyClassifier(strategy='most_frequent'), x_train, y_train, cv=kfold, scoring='accuracy')
print("accuracy for stratified classifier: " , pred.mean()*100, "%")

kfold = model_selection.KFold(n_splits=10, random_state=50)
pred = model_selection.cross_val_score(SVC(), x_train, y_train, cv=kfold, scoring='accuracy')
print("accuracy for SVM: " , pred.mean()*100, "%")

kfold = model_selection.KFold(n_splits=10, random_state=50)
pred = model_selection.cross_val_score(LogisticRegression(), x_train, y_train, cv=kfold, scoring='accuracy')
print("accuracy for Logistic Regression: " , pred.mean()*100, "%")

kfold = model_selection.KFold(n_splits=10, random_state=50)
pred = model_selection.cross_val_score(KNeighborsClassifier(), x_train, y_train, cv=kfold, scoring='accuracy')
print("accuracy for KNN: " , pred.mean()*100, "%\n")

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("Selcting Logistic Regression as our model because it achieved the highest accuracy...")
print("Tuning hyperparameters on the validation data..\n")

logistic = LogisticRegression()
C = np.logspace(0, 4, 5)
multiclass=['ovr','multinomial']
solver=['newton-cg','lbfgs','saga']
hyperparameters = dict(C=C, multi_class=multiclass, solver=solver)
GS = GridSearchCV(logistic, hyperparameters, cv=10, verbose=0)
model = GS.fit(x_train, y_train)
pred_dev = model.predict(x_dev)
print("Accuracy on the development data: ",accuracy_score(y_dev, pred_dev)*100, "%\n")

params = model.best_estimator_.get_params()
print("best parameters found through GridSearch: ",params)

print("\nMaking predictions on the test data...\n")
model = LogisticRegression(C=params['C'], multi_class=params['multi_class'], penalty=params['penalty'], solver=params['solver'])
model.fit(x_train, y_train)
pred_test = model.predict(x_test)
print("Accuracy on the test data: ",accuracy_score(y_test, pred_test)*100, "%")

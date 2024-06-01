
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from consts import K,METRIC


def read_file(file_name):
    df = pd.read_csv(file_name)
    x = df.drop(columns=['species'])
    y = df['species']

    x = np.array(x)
    y = np.array(y)
    return x,y


def create_model(x_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=K, metric=METRIC)
    knn.fit(x_train, y_train)

    return knn


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return {'accuracy': accuracy,'f1': f1 }
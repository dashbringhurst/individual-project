import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import explore
import wrangle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

def tree_model(X_train, y_train, X_validate, y_validate, depth):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=217)
    clf = clf.fit(X_train, y_train)
    plt.figure(figsize=(13, 7))
    plot_tree(clf)
    plt.show()
    
    y_pred = clf.predict(X_train)
    y_pred_proba = clf.predict_proba(X_train)
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print(classification_report(y_train, y_pred))
    print('Accuracy of Decision Tree classifier on validate set: {:.2f}'.format(clf.score(X_validate, y_validate)))

def rand_forest(X_train, y_train, depth, samples):
    # Make the model
    forest = RandomForestClassifier(max_depth=depth, min_samples_leaf=samples, random_state=217)
    # Fit the model (on train and only train)
    forest.fit(X_train, y_train)
    # Use the model
    # We'll evaluate the model's performance on train, first
    y_predictions = forest.predict(X_train)
    # Produce the classification report on the actual y values and this model's predicted y values
    report = classification_report(y_train, y_predictions, output_dict=True)
    print(f'Tree depth: {depth}, minimum sample size: {samples}')
    return pd.DataFrame(report)

#Материалы
Загрузим необходимые данные. Этот скрипт лишь предсказывает, скрипт обучения находится в немного другом файлике :) [см. описание на GitHub].
"""

!wget https://test.deqstudio.com/data1.zip -P datas/
!unzip datas/data1.zip -d datas/

print("Укажите относительный путь до файла .csv, который необходимо заполнить. Если файл находится в той же директории, что и этот скрипт, укажите его название:")
path = input()

"""Теперь необходимо загрузить нужные библиотеки и утилиты:"""

# Commented out IPython magic to ensure Python compatibility.
!pip install pandas numpy keras tensorflow

#Стандартные библиотеки для аналитики данных:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, skew
from scipy import stats
import seaborn as sns
import statsmodels.api as sm

# sklearn модули для предпроцессинга данных:
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#sklearn модули для Model Selection: 
from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#sklearn модули для эволюции и улучшения моделей:
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import make_scorer, recall_score, log_loss
from sklearn.metrics import average_precision_score

#Стандартные библиотки для визуализации данных:
import seaborn as sn
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib 
# %matplotlib inline
color = sn.color_palette()
import matplotlib.ticker as mtick
from IPython.display import display
pd.options.display.max_columns = None
from pandas.plotting import scatter_matrix
from sklearn.metrics import roc_curve

#Различные библиотки утилит:
import random
import os
import re
import sys
import timeit
import string
import time
from datetime import datetime
from time import time
from dateutil.parser import parse
import joblib

test0 = pd.read_csv(path)
test0

final_results = pd.read_csv('const/final_results.csv')
final_results

datas_predictionstable = pd.read_csv('const/datas_predictionstable.csv')
lr_classifier = RandomForestClassifier(n_estimators = 72, criterion = 'entropy', random_state = 0)

with open("const/model_complete.pkl", "rb") as f:
    model = pickle.load(f)

yPred = model.predict(datas_predictionstable)

count_row = test0.shape[0]
pd.options.mode.chained_assignment
for x in range(test0.shape[0]):
  lookfor = test0.iloc[x].contract_id
  prediction = final_results.loc[final_results['contract_id'] == int(lookfor)]
  test0.at[x, 'blocked'] = int(yPred[x])
test0.blocked = test0.blocked.astype(int)
test0

test0.to_csv(path, encoding='utf-8')

print("ОК - Записали предсказания в предоставленный файл.")

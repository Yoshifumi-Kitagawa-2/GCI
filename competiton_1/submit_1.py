#必要なライブラリをインポート
from google.colab import drive
drive.mount('/content/drive')
import os

import numpy as np
import numpy.random as random

import pandas as pd
from pandas import DataFrame

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping

#データ読み込み
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCI/コンペ1/コンペ/data/train.csv")

#PassengerId、名前、チケット番号、　キャビン番号、乗船した港は関係なさそうなのでまずはそれを排除
data_1 = data[['Perished', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

#性別をダミー変数化
dummy_sex = pd.get_dummies(data_1['Sex'], prefix='Is')
data_2 = pd.concat([data_1, dummy_sex], axis=1)

#SibSpとParchの和を取る
data_2['Familynum'] = data_2['SibSp'] + data_2['Parch']

#不要な列を削除する
data_3 = data_2[['Perished', 'Pclass', 'Age', 'Familynum', 'Fare',  'Is_female', 'Is_male']]

#欠損値の確認
missing_values = data_3.isnull().sum()
#print(missing_values) Ageだけnullがあることが分かった

#年齢だけ中央値埋める
median_age = data_3['Age'].median()
data_3['Age'].fillna(median_age, inplace=True)

#valuesだけ取り出す
data_3_values=data_3.values
#print(data_3_values)

#min_max_scalerで正規化する
scaler = MinMaxScaler()
data_3_values_scaled=scaler.fit_transform(data_3_values)

x_scaled=data_3_values_scaled[:, 1:]
y_scaled=data_3_values_scaled[:, 0]

#トレーニングとテスト分割
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

#one-hotエンコーディング
y_train_c = keras.utils.to_categorical(y_train, 2)
y_test_c = keras.utils.to_categorical(y_test, 2)

#モデルの構造
model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(6,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#モデル学習
early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1)

model.fit(x_train, y_train_c,
          epochs=100,
          batch_size=32,
          callbacks=[early_stop],
          verbose=2,
          validation_data=(x_test, y_test_c))

#predicted = np.argmax(model.predict(x_test, verbose=0), axis=-1)
#expected = y_test

#print(y_test)

#from sklearn import metrics
#print(metrics.confusion_matrix(expected, predicted))
#print(metrics.classification_report(expected, predicted))

#========================本番テスト============================
x_test_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCI/コンペ1/コンペ/data/test.csv")

x_test_date_1 = x_test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
dummy_sex_2 = pd.get_dummies(x_test_date_1['Sex'], prefix='Is')
x_test_date_2 = pd.concat([x_test_date_1, dummy_sex_2], axis=1)

#SibSpとParchの和を取る
x_test_date_2['Familynum'] = x_test_date_2['SibSp'] + x_test_date_2['Parch']

#不要な列を削除する
x_test_date_3 = x_test_date_2[['Pclass', 'Age', 'Familynum', 'Fare',  'Is_female', 'Is_male']]

#年齢だけ中央値埋める
median_age_2 = x_test_date_3['Age'].median()
x_test_date_3['Age'].fillna(median_age_2, inplace=True)

#valuesだけ取り出す
x_test_date_3_values=x_test_date_3.values

#min_max_scalerで正規化する
scaler_2 = MinMaxScaler()
x_test_date_3_scaled=scaler_2.fit_transform(x_test_date_3_values)


predicted = np.argmax(model.predict(x_test_date_3_scaled, verbose=0), axis=-1)
submission = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCI/コンペ1/コンペ/data/gender_submission_2.csv")
submission['Perished'] = predicted

# Google Drive・Google Colaboratoryで作業する場合
from google.colab import files
# colaboratory上に保存
# 保存したcsvファイルはランタイムが終了すると削除されます
submission.to_csv('submission_9.csv', index=False)
# colaboratory上に保存したcsvファイルをローカルに保存
files.download('submission_9.csv')

#====================================
#各特徴量と結果の決定係数を求める

# 特徴量と目的変数の取得
#X = data_3_values_scaled[:, 1:]  # 特徴量（Age以降の列）
#y = data_3_values_scaled[:, 0]   # 目的変数（Perished列）

# 線形回帰モデルの初期化と学習
#model = LinearRegression()
#model.fit(X, y)

# 各特徴量の決定係数を表示
#feature_names = data_3.columns[1:]
#coefficients = model.coef_
#for feature, coef in zip(feature_names, coefficients):
    #print(f'{feature}: {coef}')
#各説明変数と目的変数の決定係数結果
#Pclass: 0.3414176708260056→下層クラスほど生き残っている
#Age: 0.46487916113672545
#SibSp: 0.34845499802692687
#Parch: 0.11461194531739896
#Fare: -0.2109990041391943→下層クラスほど生き残っている。Pclassと同じ。では削除してもいいかも？
#Is_female: -0.25622549965531183
#Is_male: 0.25622549965531183
#======================================
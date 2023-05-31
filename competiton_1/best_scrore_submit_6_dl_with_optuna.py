#5_30 2回目 自己ベスト：0.822
#~5 30 21:00
#https://qiita.com/jun40vn/items/d8a1f71fae680589e05cを参考に色々改良していく
#昨日提出したDLをoptunaで最適化したい
#層4つ

#データ読み込み
train_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCI/コンペ1/コンペ/data/train.csv")
x_test_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCI/コンペ1/コンペ/data/test.csv")

#train_dataとtest_dataの連結
x_test_data['Perished'] = np.nan
join_data = pd.concat([train_data, x_test_data], ignore_index=True, sort=False)

#==========Ageの欠陥値を埋める：参照した==============
# Age を Pclass, Sex, Parch, SibSp からランダムフォレストで推定
from sklearn.ensemble import RandomForestRegressor
# 推定に使用する項目を指定
age_join_data = join_data[['Age', 'Pclass','Sex','Parch','SibSp']]

# ラベル特徴量をワンホットエンコーディング
age_join_data=pd.get_dummies(age_join_data)

# 学習データとテストデータに分離し、numpyに変換
known_age = age_join_data[age_join_data.Age.notnull()].values  
unknown_age = age_join_data[age_join_data.Age.isnull()].values

# 学習データをX, yに分離
X = known_age[:, 1:]  
y = known_age[:, 0]

# ランダムフォレストで推定モデルを構築
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)

# 推定モデルを使って、テストデータのAgeを予測し、補完
predictedAges = rfr.predict(unknown_age[:, 1::])
join_data.loc[(join_data.Age.isnull()), 'Age'] = predictedAges #補完完了!

#＝＝＝＝＝＝＝＝Nameから新たな特徴量を作り出す================
# Nameから敬称(Title)を抽出し、グルーピング#Mrの生存率が最も低く、Mrsの生存率が最も高いことが分かる
join_data['Title'] = join_data['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
join_data['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
join_data['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
join_data['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
join_data['Title'].replace(['Mlle'], 'Miss', inplace=True)
join_data['Title'].replace(['Jonkheer'], 'Master', inplace=True)
#sns.barplot(x='Title', y='Perished', data=join_data, palette='Set3')

#===============Fareの欠損値の補完===========
fare=join_data.loc[(join_data['Embarked'] == 'S') & (join_data['Pclass'] == 3), 'Fare'].median()
join_data['Fare']=join_data['Fare'].fillna(fare)

# ===============Family==================
# Family = SibSp + Parch + 1 を特徴量とし、グルーピング
join_data['Family']=join_data['SibSp']+join_data['Parch']+1
join_data.loc[(join_data['Family']>=2) & (join_data['Family']<=4), 'Family_label'] = 2
join_data.loc[(join_data['Family']>=5) & (join_data['Family']<=7) | (join_data['Family']==1), 'Family_label'] = 1  # == に注意
join_data.loc[(join_data['Family']>=8), 'Family_label'] = 0

# ＝＝＝＝＝＝＝＝Embarked＝＝＝＝＝＝＝＝
# 欠損値をSで補完
join_data['Embarked'] = join_data['Embarked'].fillna('S') 

#print(join_data.tail())

#＝＝＝＝＝＝＝＝前処理＝＝＝＝＝＝＝＝
#名前の代わりにTitle/SIbSpとParchの代わりにFamily_label/cabin、チケット番号は抜く
join_data = join_data[['Perished','Pclass','Sex','Age','Fare','Embarked','Title','Family_label']]

# ラベル特徴量をワンホットエンコーディング
join_data = pd.get_dummies(join_data)

# データセットを trainとtestに分割
train = join_data[join_data['Perished'].notnull()] #トレーニング用
test = join_data[join_data['Perished'].isnull()].drop('Perished',axis=1)

#~train, testデータそれぞれを正規化する~
#valuesだけ取り出す
train_values=train.values
test_values=test.values

#min_max_scalerでtrain, testをそれぞれ正規化する
scaler = MinMaxScaler()
train_values_scaled=scaler.fit_transform(train_values)
test_values_scaled=scaler.fit_transform(test_values) #本番用のテストで突っ込む！いじらない!

#以下は自分でテストする用 #DLバージョン
#事前に予測精度を図るためにトレーニングデータをさらにトレーニング用とテスト用に分割
train_x_scaled=train_values_scaled[:, 1:]
train_y_scaled=train_values_scaled[:, 0]
x_train, x_test, y_train, y_test = train_test_split(train_x_scaled, train_y_scaled, test_size=0.2, random_state=42)

#one-hotエンコーディング
y_train_c = keras.utils.to_categorical(y_train, 2)
y_test_c = keras.utils.to_categorical(y_test, 2)

'''
#モデルの構造
model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(15,)))
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

predicted = np.argmax(model.predict(x_test, verbose=0), axis=-1)
expected = y_test

from sklearn import metrics
print(metrics.confusion_matrix(expected, predicted))
print(metrics.classification_report(expected, predicted))
'''

def objective(trial):
    # ハイパーパラメータの探索範囲を定義
    units = trial.suggest_int('units', 10, 100)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    # モデルの構築
    model = Sequential()
    model.add(Dense(units, activation='relu', input_shape=(15,)))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # モデルの学習
    model.fit(x_train, y_train_c, epochs=100, batch_size=32, verbose=0)

    # 評価指標（正解率など）を取得
    score = model.evaluate(x_test, y_test_c, verbose=0)

    # 目的関数は最小化するため、正解率の逆数を返す
    return 1.0 - score[1]

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

best_params = study.best_params

best_units = best_params['units']
best_learning_rate = best_params['learning_rate']

model = Sequential()
model.add(Dense(best_units, activation='relu', input_shape=(15,)))
model.add(Dense(best_units, activation='relu'))
model.add(Dense(best_units, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# データとラベルを分割
X = train_values_scaled[:, 1:]
y = train_values_scaled[:, 0]

# K-fold交差検証の設定
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

# 各foldでモデルを学習・評価
for train_index, val_index in kfold.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # one-hotエンコーディング
    y_train_c = keras.utils.to_categorical(y_train, 2)
    y_val_c = keras.utils.to_categorical(y_val, 2)

    # モデルの学習
    model.fit(X_train, y_train_c, epochs=100, batch_size=32, verbose=0)

    # モデルの評価
    score = model.evaluate(X_val, y_val_c, verbose=0)
    scores.append(score[1])

# 交差検証の結果を表示
print('Cross Validation Accuracy:', np.mean(scores))



predicted = np.argmax(model.predict(test_values_scaled, verbose=0), axis=-1)
submission = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCI/コンペ1/コンペ/data/gender_submission_2.csv")
submission['Perished'] = predicted

# Google Drive・Google Colaboratoryで作業する場合
from google.colab import files
# colaboratory上に保存
# 保存したcsvファイルはランタイムが終了すると削除されます
submission.to_csv('submission_g.csv', index=False)
# colaboratory上に保存したcsvファイルをローカルに保存
files.download('submission_g.csv')
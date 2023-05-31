#5_30 1回目
#~5 30 15:00
#https://qiita.com/jun40vn/items/d8a1f71fae680589e05cを参考に色々改良していく
#ランダムフォレストバージョン

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
#join_data = join_data[['Perished','Pclass','Sex','Age','Fare','Title','Family_label']]

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

rfc = RandomForestClassifier(random_state=0)
rfc.fit(x_train, y_train)
print('='*20)
print('RandomForestClassifier')
print(f'accuracy of train set: {rfc.score(x_train, y_train)}')
print(f'accuracy of test set: {rfc.score(x_test, y_test)}')



# CV分割数
cv = 5
def objective(trial):
    
    param_grid_rfc = {
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        'min_samples_split': trial.suggest_int("min_samples_split", 7, 15),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        'max_features': trial.suggest_int("max_features", 3, 10),
        "random_state": 0
    }

    model = RandomForestClassifier(**param_grid_rfc)
    
    # 5-Fold CV / Accuracy でモデルを評価する
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=x_train, y=y_train, cv=kf)
    # 最小化なので 1.0 からスコアを引く
    return scores['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)
print(study.best_value)
rfc_best_param = study.best_params

# 5-Fold CV / Accuracy でモデルを評価する
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

rfc_best = RandomForestClassifier(**rfc_best_param)
print('RandomForestClassifier')
print('='*20)
scores = cross_validate(rfc_best, X=train_x_scaled, y=train_y_scaled, cv=kf)
print(f'mean:{scores["test_score"].mean()}, std:{scores["test_score"].std()}')
print('='*20)

svc_best_submit = SVC(**svc_best_param)
svc_best_submit.fit(train_x_scaled, train_y_scaled)
# テストデータを予測
predictions = svc_best_submit.predict(test_values_scaled)
# 予測結果を表示
print(predictions)

predictions = predictions.astype(int)
submission = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCI/コンペ1/コンペ/data/gender_submission_2.csv")
submission['Perished'] = predictions


# Google Drive・Google Colaboratoryで作業する場合
from google.colab import files
# colaboratory上に保存
# 保存したcsvファイルはランタイムが終了すると削除されます
submission.to_csv('submission_f.csv', index=False)
# colaboratory上に保存したcsvファイルをローカルに保存
files.download('submission_f.csv')
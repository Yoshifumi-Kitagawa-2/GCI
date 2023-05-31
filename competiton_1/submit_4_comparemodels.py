#5_29 3回目
#~5 30 3:00
#https://qiita.com/jun40vn/items/d8a1f71fae680589e05cを参考に色々改良していく
#色んなモデルで試す

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

# =================Surname==================
# NameからSurname(苗字)を抽出
#join_data['Surname'] = join_data['Name'].map(lambda name:name.split(',')[0].strip())

# 同じSurname(苗字)の出現頻度をカウント(出現回数が2以上なら家族)
#join_data['FamilyGroup'] = join_data['Surname'].map(join_data['Surname'].value_counts())

# 家族で16才以下または女性の生存率
#Female_Child_Group=join_data.loc[(join_data['FamilyGroup']>=2) & ((join_data['Age']<=16) | (join_data['Sex']=='female'))]
#Female_Child_Group=Female_Child_Group.groupby('Surname')['Perished'].mean()

# 家族で16才超えかつ男性の生存率
#Male_Adult_Group=join_data.loc[(join_data['FamilyGroup']>=2) & (join_data['Age']>16) & (join_data['Sex']=='male')]
#Male_Adult_List=Male_Adult_Group.groupby('Surname')['Perished'].mean()

# デッドリストとサバイブリストの作成
#Dead_list=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
#Survived_list=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)

# デッドリストとサバイブリストの表示
#print('Dead_list = ', Dead_list)
#print('Survived_list = ', Survived_list)

# デッドリストとサバイブリストをSex, Age, Title に反映させる
#join_data.loc[(join_data['Perished'].isnull()) & (join_data['Surname'].apply(lambda x:x in Dead_list)),['Sex','Age','Title']] = ['male',28.0,'Mr']
#join_data.loc[(join_data['Perished'].isnull()) & (join_data['Surname'].apply(lambda x:x in Survived_list)),['Sex','Age','Title']] = ['female',5.0,'Mrs']

#===============Fareの欠損値の補完===========
fare=join_data.loc[(join_data['Embarked'] == 'S') & (join_data['Pclass'] == 3), 'Fare'].median()
join_data['Fare']=join_data['Fare'].fillna(fare)

# ===============Family==================
# Family = SibSp + Parch + 1 を特徴量とし、グルーピング
join_data['Family']=join_data['SibSp']+join_data['Parch']+1
join_data.loc[(join_data['Family']>=2) & (join_data['Family']<=4), 'Family_label'] = 2
join_data.loc[(join_data['Family']>=5) & (join_data['Family']<=7) | (join_data['Family']==1), 'Family_label'] = 1  # == に注意
join_data.loc[(join_data['Family']>=8), 'Family_label'] = 0

#===============Ticketから意味のある特徴量を取り出す============
# 同一Ticketナンバーの人が何人いるかを特徴量として抽出
#Ticket_Count = dict(join_data['Ticket'].value_counts())
#join_data['TicketGroup'] = join_data['Ticket'].map(Ticket_Count)
#sns.barplot(x='TicketGroup', y='Survived', data=df, palette='Set3')
#plt.show()
#join_data.loc[(join_data['TicketGroup']>=2) & (join_data['TicketGroup']<=4), 'Ticket_label'] = 2
#join_data.loc[(join_data['TicketGroup']>=5) & (join_data['TicketGroup']<=8) | (join_data['TicketGroup']==1), 'Ticket_label'] = 1  
#join_data.loc[(join_data['TicketGroup']>=11), 'Ticket_label'] = 0

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

xgb = XGBClassifier(random_state=0)
xgb.fit(x_train, y_train)
print('='*20)
print('XGBClassifier')
print(f'accuracy of train set: {xgb.score(x_train, y_train)}')
print(f'accuracy of train set: {xgb.score(x_test, y_test)}')

lgb = LGBMClassifier(random_state=0)
lgb.fit(x_train, y_train)
print('='*20)
print('LGBMClassifier')
print(f'accuracy of train set: {lgb.score(x_train, y_train)}')
print(f'accuracy of train set: {lgb.score(x_test, y_test)}')

lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)
print('='*20)
print('LogisticRegression')
print(f'accuracy of train set: {lr.score(x_train, y_train)}')
print(f'accuracy of train set: {lr.score(x_test, y_test)}')

svc = SVC(random_state=0)
svc.fit(x_train, y_train)
print('='*20)
print('SVC')
print(f'accuracy of train set: {svc.score(x_train, y_train)}')
print(f'accuracy of train set: {svc.score(x_test, y_test)}')

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

def objective(trial):
    
    param_grid_xgb = {
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 5),
        'gamma': trial.suggest_discrete_uniform("gamma", 0.1, 1.0, 0.1),
        'subsample': trial.suggest_discrete_uniform("subsample", 0.5, 1.0, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform("colsample_bytree", 0.5, 1.0, 0.1),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        "random_state": 0
    }

    model = XGBClassifier(**param_grid_xgb)
    
    # 5-Fold CV / Accuracy でモデルを評価する
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=x_train, y=y_train, cv=kf)
    # 最小化なので 1.0 からスコアを引く
    return scores['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)
print(study.best_value)
xgb_best_param = study.best_params

def objective(trial):
    
    param_grid_lgb = {
        'num_leaves': trial.suggest_int("num_leaves", 3, 10),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        "random_state": 0
    }

    model = LGBMClassifier(**param_grid_lgb)
    
    # 5-Fold CV / Accuracy でモデルを評価する
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=x_train, y=y_train, cv=kf)
    # 最小化なので 1.0 からスコアを引く
    return scores['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)
print(study.best_value)
lgb_best_param = study.best_params

import warnings
warnings.filterwarnings('ignore')

def objective(trial):
    
    param_grid_lr = {
        'C' : trial.suggest_int("C", 1, 100),
        "random_state": 0
    }

    model = LogisticRegression(**param_grid_lr)
    
    # 5-Fold CV / Accuracy でモデルを評価する
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=x_train, y=y_train, cv=kf)
    # 最小化なので 1.0 からスコアを引く
    return scores['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
print(study.best_params)
print(study.best_value)
lr_best_param = study.best_params

import warnings
warnings.filterwarnings('ignore')

def objective(trial):
    
    param_grid_svc = {
        'C' : trial.suggest_int("C", 50, 200),
        'gamma': trial.suggest_loguniform("gamma", 1e-4, 1.0),
        "random_state": 0,
        'kernel': 'rbf'
    }

    model = SVC(**param_grid_svc)
    
    # 5-Fold CV / Accuracy でモデルを評価する
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=x_train, y=y_train, cv=kf)
    # 最小化なので 1.0 からスコアを引く
    return scores['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
print(study.best_params)
print(study.best_value)
svc_best_param = study.best_params

# 5-Fold CV / Accuracy でモデルを評価する
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

rfc_best = RandomForestClassifier(**rfc_best_param)
print('RandomForestClassifier')
print('='*20)
scores = cross_validate(rfc_best, X=train_x_scaled, y=train_y_scaled, cv=kf)
print(f'mean:{scores["test_score"].mean()}, std:{scores["test_score"].std()}')
print('='*20)

xgb_best = XGBClassifier(**xgb_best_param)
print('XGBClassifier')
print('='*20)
scores = cross_validate(xgb_best, X=train_x_scaled, y=train_y_scaled, cv=kf)
print(f'mean:{scores["test_score"].mean()}, std:{scores["test_score"].std()}')
print('='*20)

lgb_best = LGBMClassifier(**lgb_best_param)
print('LGBMClassifier')
print('='*20)
scores = cross_validate(lgb_best, X=train_x_scaled, y=train_y_scaled, cv=kf)
print(f'mean:{scores["test_score"].mean()}, std:{scores["test_score"].std()}')
print('='*20)

lr_best = LogisticRegression(**lr_best_param)
print('LogisticRegression')
print('='*20)
scores = cross_validate(lr_best, X=train_x_scaled, y=train_y_scaled, cv=kf)
print(f'mean:{scores["test_score"].mean()}, std:{scores["test_score"].std()}')
print('='*20)

svc_best = SVC(**svc_best_param)
print('SVC')
print('='*20)
scores = cross_validate(svc_best, X=train_x_scaled, y=train_y_scaled, cv=kf)
print(f'mean:{scores["test_score"].mean()}, std:{scores["test_score"].std()}')
print('='*20)
'''
#one-hotエンコーディング
y_train_c = keras.utils.to_categorical(y_train, 2)
y_test_c = keras.utils.to_categorical(y_test, 2)

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

'''
predicted = np.argmax(model.predict(test_values_scaled, verbose=0), axis=-1)
submission = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/GCI/コンペ1/コンペ/data/gender_submission_2.csv")
submission['Perished'] = predicted
'''

# Google Drive・Google Colaboratoryで作業する場合
#from google.colab import files
# colaboratory上に保存
# 保存したcsvファイルはランタイムが終了すると削除されます
#submission.to_csv('submission_e.csv', index=False)
# colaboratory上に保存したcsvファイルをローカルに保存
#files.download('submission_e.csv')
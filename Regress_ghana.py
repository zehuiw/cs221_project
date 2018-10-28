import numpy as np
# import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets,linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
selected_feature_num = 43 #31
selected_feature_index=[4, 32, 36, 11, 28, 26, 15, 27, 38, 18, 17, 33, 31, 9, 23, 20, 13, 24, 34, 21, 16,  2 ,25, 42 ,12,
  3, 37, 39  ,7 ,41 ,30 ,14, 22 , 1 ,29 , 6 ,35  ,0 , 5 ,40 ,10 ,19, 8]
#read dataset
data_path='dataset/experiments/PPI_all_score_size_distance.txt'
with open(data_path) as f:
    lines=f.readlines()
    ID=np.empty([len(lines),1], dtype=int)
    ppi=np.empty([len(lines),1], dtype=int)
    feature=np.empty([len(lines), selected_feature_num], dtype=float)
    cnt=0
    for line in lines:
        t_info=line.strip('\n').split(',')
        if t_info[0]=='ID':
            feature_names=np.array(t_info[1:44])
            continue
        ID[cnt]=(int(t_info[0]))
        ppi[cnt]=(int(t_info[-1]))
        feature[cnt]=np.transpose(np.array([t_info[i] for i in selected_feature_index[0:selected_feature_num]]))
        cnt+=1
#split train and test

ids,X,y=shuffle(ID,feature,ppi,random_state=0)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
id_train,id_test=ids[:offset],ids[offset:]

####################Normalization####################################
#min-max normalization

X_max=X_train.max(axis=0)
X_min=X_train.min(axis=0)
X_tmax=np.tile(X_max,(len(X_train),1))
X_tmin=np.tile(X_min,(len(X_train),1))
# print (feature[0, :])
# print (X_train[0, :])
# print (X_min, X_max)
X_train=(X_train-X_tmin)/(X_tmax-X_tmin)

X_tmax=np.tile(X_max,(len(X_test),1))
X_tmin=np.tile(X_min,(len(X_test),1))
X_test=(X_test-X_tmin)/(X_tmax-X_tmin)
# y_max=y.max(axis=0)
# y_min=y.min(axis=0)
#
# y_tmax=np.tile(y_max,(len(y_train),1))
# y_tmin=np.tile(y_min,(len(y_train),1))
# y_train=(y_train-y_tmin)/(y_tmax-y_tmin)
# y_tmax=np.tile(y_max,(len(y_test),1))
# y_tmin=np.tile(y_min,(len(y_test),1))
# y_test=(y_test-y_tmin)/(y_tmax-y_tmin)


#gaussian normalization
# X_mean=X_train.mean(axis=0)
# X_var=X_train.var(axis=0)
# X_tmean=np.tile(X_mean,(len(X_train),1))
# X_tvar=np.tile(X_var,(len(X_train),1))
# X_train=(X_train-X_tmean)/X_tvar+1e-10
#
# X_tmean=np.tile(X_mean,(len(X_test),1))
# X_tvar=np.tile(X_var,(len(X_test),1))
# X_test=(X_test-X_tmean)/X_tvar+1e-10
# y_max=y.max(axis=0)
# y_min=y.min(axis=0)
#
# y_tmax=np.tile(y_max,(len(y_train),1))
# y_tmin=np.tile(y_min,(len(y_train),1))
# y_train=(y_train-y_tmin)/(y_tmax-y_tmin)
# y_tmax=np.tile(y_max,(len(y_test),1))
# y_tmin=np.tile(y_min,(len(y_test),1))
# y_test=(y_test-y_tmin)/(y_tmax-y_tmin)
######################################################################


####################linear Models####################################
#
# regr = linear_model.Ridge (alpha = .1)
# regr.fit(X_train, y_train)
# y_pred = regr.predict(X_test)
# print('Coefficients: \n', regr.coef_)

######################################################################

####################Random  Forest####################################

# regr = RandomForestRegressor(max_depth=2, random_state=0,
#                             n_estimators=50000)
# regr.fit(X_train, np.ravel(y_train,order='C'))
# y_pred = regr.predict(X_test)

######################################################################

####################GBDT Tree#########################################

params = {'n_estimators': 6000, 'max_depth': 2, 'min_samples_split': 2,
          'learning_rate': 0.005, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, np.ravel(y_train,order='C'))
y_pred = clf.predict(X_test)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
#
# for i, y_pred in enumerate(clf.staged_predict(X_test)):
#     test_score[i] = clf.loss_(y_test, y_pred)
#
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
#          label='Training Set Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
#          label='Test Set Deviance')
# plt.legend(loc='upper right')
# plt.xlabel('Boosting Iterations')
# plt.ylabel('Deviance')
# plt.show()
#####################################################################

def output():
    # show results
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    # Plot outputs
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test,y_pred,  color='black')
    plt.plot(np.sort(y_test),np.sort(y_test), color='blue', linewidth=3)
    plt.xlim((np.min(y_pred),np.max(y_pred)))
    plt.xlim((np.min(y_test),np.max(y_test)))
    plt.xlabel('gt')
    plt.ylabel('prediction')
    plt.xticks(())
    plt.yticks(())
    plt.title('R2 %.2f'% r2_score(y_test, y_pred))
    # plt.show()

    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
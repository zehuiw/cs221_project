
from sklearn import ensemble
from sklearn import datasets,linear_model
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#read dataset
data_path='dataset\\experiments\\PPI_all_score.txt'
with open(data_path) as f:
    lines=f.readlines()
    ID=np.empty([len(lines),1], dtype=int)
    ppi=np.empty([len(lines),1], dtype=int)
    feature=np.empty([len(lines), len(lines[0].strip('\n').split(','))-2], dtype=float)
    cnt=0
    for line in lines:
        t_info=line.strip('\n').split(',')
        if t_info[0]=='ID':
            continue
        ID[cnt]=(int(t_info[0]))
        ppi[cnt]=(int(t_info[-1]))
        feature[cnt]=np.transpose(np.array(t_info[1:44]))
        cnt+=1

#split train and test

ids,X,y=shuffle(ID,feature,ppi,random_state=0)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
id_train,id_test=ids[:offset],ids[offset:]

####################Normalization####################################
#min-max normalization

X_max=X_train.max(axis=0)
X_min=X_train.min(axis=0)
X_tmax=np.tile(X_max,(len(X_train),1))
X_tmin=np.tile(X_min,(len(X_train),1))
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

r2_tmp=0
n_est=0
lr=0
max_d=0
####################GBDT Tree#########################################
for ne in range(1000,10000,1000):
    for l in [0.1,0.05,0.01,0.005,0.001]:
        for md in range(2, 3, 1):
            # print(md)
            params = {'n_estimators': ne, 'max_depth': md, 'min_samples_split': 2,
                      'learning_rate': l, 'loss': 'ls'}
            clf = ensemble.GradientBoostingRegressor(**params)

            clf.fit(X_train, np.ravel(y_train,order='C'))
            y_pred = clf.predict(X_test)
            if r2_score(y_test, y_pred)>r2_tmp:
                n_est=ne
                lr=l
                max_d=md
                r2_tmp=r2_score(y_test, y_pred)
                print(n_est)
                print(lr)
                print(max_d)
                print("Mean squared error: %.2f"
                      % mean_squared_error(y_test, y_pred))
                print('Variance score: %.2f' % r2_score(y_test, y_pred))

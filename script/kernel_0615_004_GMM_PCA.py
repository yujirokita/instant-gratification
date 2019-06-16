"""
- GMM with n_components=1..4
- y-flip
- pseudo labeling
- stacking (logistic regression) with CV
- PCA
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# parameters
MAX_COMPONENTS = 4
RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)
N_SPLITS = 10; kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
MAX_MAGIC_NO = 512

# load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
ids_tr = train.pop('id').values; ids_te = test.pop('id').values
magic_tr = train.pop('wheezy-copper-turtle-magic').values; magic_te = test.pop('wheezy-copper-turtle-magic').values
target = train.pop('target').values
train = train.values; test = test.values
    
# infomative columns of each magic value
vt = VarianceThreshold(threshold=1.5)
infomative_cols = []
for i in range(MAX_MAGIC_NO):
    vt.fit(train[magic_tr==i])
    infomative_cols.append(vt.get_support(indices=True))

### Step-1 ###
oof_all = []
pred_all = []
for n in range(1, MAX_COMPONENTS+1):
    oof_n = np.zeros(len(train))
    pred_n = np.zeros(len(test))
    gmm0 = GaussianMixture(n_components=n, covariance_type='full', random_state=RANDOM_SEED)
    gmm1 = GaussianMixture(n_components=n, covariance_type='full', random_state=RANDOM_SEED)
    for i in range(MAX_MAGIC_NO):
        print('.', end='')
        train_i = train[magic_tr==i][:,infomative_cols[i]]
        target_i = target[magic_tr==i]
        test_i = test[magic_te==i][:,infomative_cols[i]]
        oof_n_i = np.zeros(len(train_i))
        pred_n_i = np.zeros(len(test_i))

        # PCA
        pca = PCA(0.99).fit(np.concatenate([train_i, test_i], axis=0))
        train_i = pca.transform(train_i)
        test_i = pca.transform(test_i)

        for trn_idx, val_idx in kfold.split(train_i, target_i):
            trn_train = train_i[trn_idx,:]
            trn_target = target_i[trn_idx]
            val_train = train_i[val_idx,:]
            gmm0.fit(trn_train[trn_target==0])
            gmm1.fit(trn_train[trn_target==1])
            oof_n_i[val_idx] = gmm1.score_samples(val_train) - gmm0.score_samples(val_train)
            pred_n_i += (gmm1.score_samples(test_i) - gmm0.score_samples(test_i)) / kfold.n_splits
        oof_n[magic_tr==i] = oof_n_i
        pred_n[magic_te==i] = pred_n_i
    oof_all.append(oof_n)
    pred_all.append(pred_n)
    print()
    print(f'SCORE (STEP-1, n_component={n}): {roc_auc_score(target, oof_n)}')

# combine predictions
oof_mat = np.stack(oof_all, axis=1)
pred_mat = np.stack(pred_all, axis=1)
oof = np.zeros(len(train))
pred = np.zeros(len(test))
for trn_idx, val_idx in kfold.split(oof_mat, target):
    comb = LogisticRegression(solver='lbfgs').fit(oof_mat[trn_idx,:], target[trn_idx])
    oof[val_idx] = comb.predict_proba(oof_mat[val_idx,:])[:,1]
    pred += comb.predict_proba(pred_mat)[:,1] / kfold.n_splits

print(f'SCORE (STEP-1): {roc_auc_score(target, oof)}')

### Step-2 ###
target2 = np.where(oof>0.5, 1, 0)
pseudo_label = np.copy(pred)

oof_all = []
pred_all = []
for n in range(1, MAX_COMPONENTS+1):
    oof_n = np.zeros(len(train))
    pred_n = np.zeros(len(test))
    gmm0 = GaussianMixture(n_components=n, covariance_type='full', random_state=RANDOM_SEED)
    gmm1 = GaussianMixture(n_components=n, covariance_type='full', random_state=RANDOM_SEED)
    for i in range(MAX_MAGIC_NO):
        print('.', end='')
        train_i = train[magic_tr==i][:,infomative_cols[i]]
        target_i = target2[magic_tr==i] # modified target
        test_i = test[magic_te==i][:,infomative_cols[i]]
        oof_n_i = np.zeros(len(train_i))
        pred_n_i = np.zeros(len(test_i))

        # PCA
        pca = PCA(0.99).fit(np.concatenate([train_i, test_i], axis=0))
        train_i = pca.transform(train_i)
        test_i = pca.transform(test_i)

        pseudo_label_i = pseudo_label[magic_te==i]
        test0_i = test_i[pseudo_label_i<0.5]
        test1_i = test_i[pseudo_label_i>0.5]

        for trn_idx, val_idx in kfold.split(train_i, target_i):
            trn_train = np.concatenate([train_i[trn_idx,:], test0_i, test1_i], axis=0)
            trn_target = np.concatenate([target_i[trn_idx], np.zeros(len(test0_i)), np.ones(len(test1_i))])
            val_train = train_i[val_idx,:]
            gmm0.fit(trn_train[trn_target==0])
            gmm1.fit(trn_train[trn_target==1])
            oof_n_i[val_idx] = gmm1.score_samples(val_train) - gmm0.score_samples(val_train)
            pred_n_i += (gmm1.score_samples(test_i) - gmm0.score_samples(test_i)) / kfold.n_splits
        oof_n[magic_tr==i] = oof_n_i
        pred_n[magic_te==i] = pred_n_i
    oof_all.append(oof_n)
    pred_all.append(pred_n)
    print()
    print(f'SCORE (STEP-2, n_component={n}): {roc_auc_score(target, oof_n)}')

# combine predictions
oof_mat = np.stack(oof_all, axis=1)
pred_mat = np.stack(pred_all, axis=1)
oof = np.zeros(len(train))
pred = np.zeros(len(test))
for trn_idx, val_idx in kfold.split(oof_mat, target):
    comb = LogisticRegression(solver='lbfgs').fit(oof_mat[trn_idx,:], target[trn_idx])
    oof[val_idx] = comb.predict_proba(oof_mat[val_idx,:])[:,1]
    pred += comb.predict_proba(pred_mat)[:,1] / kfold.n_splits

print(f'SCORE (STEP-2): {roc_auc_score(target, oof)}')

# submission
sub =  pd.DataFrame.from_dict({'id': ids_te, 'target': pred})
sub.to_csv('submission.csv', index=False)
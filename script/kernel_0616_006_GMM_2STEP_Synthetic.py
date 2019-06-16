"""
- STEP1
    - GMM (n_components=1..4)
- STEP2
    - GMM (n_components=1..4)
    - y-flip
    - pseudo labeling
    - synthetic data
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression

# parameters
RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)
N_SPLITS = 10; kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
MAX_MAGIC_NO = 512

class GMMBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_clusters=1, random_state=None):
        self.n_clusters_list = list(range(1, max_clusters+1))
        self.random_state = random_state

    def fit(self, X, y):
        """
        y must be composed of 0 and 1
        """
        self.gmms_ = {0:[], 1:[]}
        ll_list = []
        for n in self.n_clusters_list:
            gmm0 = GaussianMixture(n_components=n, covariance_type='full', random_state=self.random_state).fit(X[y==0])
            gmm1 = GaussianMixture(n_components=n, covariance_type='full', random_state=self.random_state).fit(X[y==1])
            self.gmms_[0].append(gmm0)
            self.gmms_[1].append(gmm1)
            ll = gmm1.score_samples(X) - gmm0.score_samples(X)
            ll_list.append(ll)

        ll_arr = np.stack(ll_list, axis=1)
        self.comb_ = LogisticRegression(solver='lbfgs').fit(ll_arr, y)
        return self

    def score_samples(self, X):
        ll_list = []
        for i, _ in enumerate(self.n_clusters_list):
            gmm0 = self.gmms_[0][i]
            gmm1 = self.gmms_[1][i]
            ll = gmm1.score_samples(X) - gmm0.score_samples(X)
            ll_list.append(ll)
        return np.stack(ll_list, axis=1)

    def sample(self, cls, n_samples_per_component):
        samples_list = [self.gmms_[cls][i].sample(n_samples_per_component)[0] for i, _ in enumerate(self.n_clusters_list)]
        return np.vstack(samples_list)

    '''
    def predict(self, X):
        ll_arr = self.score_samples(X)
        return self.comb_.predict(ll_arr)

    def predict_proba(self, X):
        ll_arr = self.score_samples(X)
        return self.comb_.predict_proba(ll_arr)
    '''


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


### STEP1 ###
max_clusters = 4

scores_tr = np.zeros((len(train), max_clusters))
scores_te = np.zeros((len(test), max_clusters))
synthetic0_list = []
synthetic1_list = []
for i in range(MAX_MAGIC_NO):
    clf = GMMBinaryClassifier(max_clusters=max_clusters, random_state=RANDOM_SEED)
    print('.', end='')
    train_i = train[magic_tr==i][:,infomative_cols[i]]
    target_i = target[magic_tr==i]
    test_i = test[magic_te==i][:,infomative_cols[i]]
    synthetic0_i_list = []
    synthetic1_i_list = []
    scores_tr_i = np.zeros((len(train_i), max_clusters))
    scores_te_i = np.zeros((len(test_i), max_clusters))
    for trn_idx, val_idx in kfold.split(train_i, target_i):
        clf.fit(train_i[trn_idx], target_i[trn_idx])
        scores_tr_i[val_idx] = clf.score_samples(train_i[val_idx])
        scores_te_i += clf.score_samples(test_i) / kfold.n_splits
        synthetic0_i_list.append(clf.sample(0, 20))
        synthetic1_i_list.append(clf.sample(1, 20))
    scores_tr[magic_tr==i] = scores_tr_i
    scores_te[magic_te==i] = scores_te_i
    synthetic0_list.append(np.vstack(synthetic0_i_list))
    synthetic1_list.append(np.vstack(synthetic1_i_list))

# combine liklihood scores
oof = np.zeros(len(train))
pred = np.zeros(len(test))
for trn_idx, val_idx in kfold.split(scores_tr, target):
    comb = LogisticRegression(solver='lbfgs').fit(scores_tr[trn_idx,:], target[trn_idx])
    oof[val_idx] = comb.predict_proba(scores_tr[val_idx,:])[:,1]
    pred += comb.predict_proba(scores_te)[:,1] / kfold.n_splits
print(f'SCORE (STEP1): {roc_auc_score(target, oof)}')


### STEP2 ###
target2 = np.where(oof>0.5, 1, 0)
pseudo_label = np.where(pred>0.5, 1, 0)

max_clusters = 4

scores_tr = np.zeros((len(train), max_clusters))
scores_te = np.zeros((len(test), max_clusters))
for i in range(MAX_MAGIC_NO):
    clf = GMMBinaryClassifier(max_clusters=max_clusters, random_state=RANDOM_SEED)
    print('.', end='')
    train_i = train[magic_tr==i][:,infomative_cols[i]]
    target_i = target2[magic_tr==i] # modified target
    test_i = test[magic_te==i][:,infomative_cols[i]]
    synthetic0_i = synthetic0_list[i]
    synthetic1_i = synthetic1_list[i]
    scores_tr_i = np.zeros((len(train_i), max_clusters))
    scores_te_i = np.zeros((len(test_i), max_clusters))
    for trn_idx, val_idx in kfold.split(train_i, target_i):
        X = np.vstack([train_i[trn_idx], synthetic0_i, synthetic1_i])
        y = np.concatenate([target_i[trn_idx], np.zeros(len(synthetic0_i)), np.ones(len(synthetic1_i))])
        clf.fit(X, y)
        scores_tr_i[val_idx] = clf.score_samples(train_i[val_idx])
        scores_te_i += clf.score_samples(test_i) / kfold.n_splits
    scores_tr[magic_tr==i] = scores_tr_i
    scores_te[magic_te==i] = scores_te_i

# combine liklihood scores
oof = np.zeros(len(train))
pred = np.zeros(len(test))
for trn_idx, val_idx in kfold.split(scores_tr, target):
    comb = LogisticRegression(solver='lbfgs').fit(scores_tr[trn_idx,:], target[trn_idx])
    oof[val_idx] = comb.predict_proba(scores_tr[val_idx,:])[:,1]
    pred += comb.predict_proba(scores_te)[:,1] / kfold.n_splits
print(f'SCORE (STEP2): {roc_auc_score(target, oof)}')


# submission
sub =  pd.DataFrame.from_dict({'id': ids_te, 'target': pred})
sub.to_csv('submission.csv', index=False)
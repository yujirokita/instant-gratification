"""
- STEP1
    - GMM (n_components=1..4)
- STEP2
    - GMM (n_components=1..4)
    - y-flip
    - pseudo labeling
    - synthetic data
- STEP3
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
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

# global parameters
RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)
N_SPLITS = 10; kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
MAX_MAGIC_NO = 512
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'

# load data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

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

# definition
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
        samples = np.vstack(samples_list)
        np.random.shuffle(samples)
        return samples

    '''
    def predict(self, X):
        ll_arr = self.score_samples(X)
        return self.comb_.predict(ll_arr)

    def predict_proba(self, X):
        ll_arr = self.score_samples(X)
        return self.comb_.predict_proba(ll_arr)
    '''


def train_score(
    train, target, magic_tr, test, magic_te,
    use_pseudo=False, pseudo_label=None,
    use_synthetic=False, synthetic0_list=None, synthetic1_list=None,
    gen_samples=True, n_samples_per_component=10
    ):

    for i in range(MAX_MAGIC_NO):
        # model
        clf = GMMBinaryClassifier(max_clusters=max_clusters, random_state=RANDOM_SEED)
        # data
        train_i = train[magic_tr==i][:,infomative_cols[i]]
        target_i = target[magic_tr==i]
        test_i = test[magic_te==i][:,infomative_cols[i]]
        if use_pseudo:
            pseudo_label_i = pseudo_label[magic_te==i]
            test0_i = test_i[pseudo_label_i==0]
            test1_i = test_i[pseudo_label_i==1]
        if use_synthetic:
            synthetic0_i = synthetic0_list[i]
            synthetic1_i = synthetic1_list[i]
        # placeholders
        scores_tr_i = np.zeros((len(train_i), max_clusters))
        scores_te_i = np.zeros((len(test_i), max_clusters))
        if gen_samples:
            samples0_i_list = []
            samples1_i_list = []

        for j, (trn_idx, val_idx) in enumerate(kfold.split(train_i, target_i)):
            X = train_i[trn_idx]
            y = target_i[trn_idx]
            if use_pseudo:
                X = np.vstack([X, test0_i, test1_i])
                y = np.concatenate([y, np.zeros(len(test0_i)), np.ones(len(test1_i))])
            if use_synthetic:
                synthetic0_i_j = synthetic0_i[len(synthetic0_i)*j//kfold.n_splits : len(synthetic0_i)*(j+1)//kfold.n_splits]
                synthetic1_i_j = synthetic1_i[len(synthetic1_i)*j//kfold.n_splits : len(synthetic1_i)*(j+1)//kfold.n_splits]
                X = np.vstack([X, synthetic0_i_j, synthetic1_i_j])
                y = np.concatenate([y, np.zeros(len(synthetic0_i_j)), np.ones(len(synthetic1_i_j))])
            clf.fit(X, y)
            scores_tr_i[val_idx] = clf.score_samples(train_i[val_idx])
            scores_te_i += clf.score_samples(test_i) / kfold.n_splits
            if gen_samples:
                samples0_i_list.append(clf.sample(0, n_samples_per_component))
                samples1_i_list.append(clf.sample(1, n_samples_per_component))
        scores_tr[magic_tr==i] = scores_tr_i
        scores_te[magic_te==i] = scores_te_i
        if gen_samples:
            samples0_list.append(np.vstack(samples0_i_list))
            samples1_list.append(np.vstack(samples1_i_list))


#################################### STEP1 ####################################
# loacl parameters
max_clusters = 4

# placeholders
scores_tr = np.zeros((len(train), max_clusters))
scores_te = np.zeros((len(test), max_clusters))
samples0_list = []
samples1_list = []

# train, score, and sample
train_score(train, target, magic_tr, test, magic_te,
    use_pseudo=False,
    use_synthetic=False,
    gen_samples=True, n_samples_per_component=100)

# combine liklihood scores
oof = np.zeros(len(train))
pred = np.zeros(len(test))
for trn_idx, val_idx in kfold.split(scores_tr, target):
    comb = LogisticRegression(solver='lbfgs').fit(scores_tr[trn_idx,:], target[trn_idx])
    oof[val_idx] = comb.predict_proba(scores_tr[val_idx,:])[:,1]
    pred += comb.predict_proba(scores_te)[:,1] / kfold.n_splits
print(f'SCORE (STEP1): {roc_auc_score(target, oof)}')


#################################### STEP2 ####################################
# loacl parameters
target_for_train = np.where(oof>0.5, 1, 0)
pseudo_label = np.where(pred>0.5, 1, 0)
synthetic0_list = deepcopy(samples0_list)
synthetic1_list = deepcopy(samples1_list)
max_clusters = 4

# placeholders
scores_tr = np.zeros((len(train), max_clusters))
scores_te = np.zeros((len(test), max_clusters))
samples0_list = []
samples1_list = []

# train, score, and sample
train_score(train, target_for_train, magic_tr, test, magic_te,
    use_pseudo=True, pseudo_label=pseudo_label,
    use_synthetic=True, synthetic0_list=synthetic0_list, synthetic1_list=synthetic1_list,
    gen_samples=True, n_samples_per_component=200)

# combine liklihood scores
oof = np.zeros(len(train))
pred = np.zeros(len(test))
for trn_idx, val_idx in kfold.split(scores_tr, target_for_train):
    X = np.vstack([scores_tr[trn_idx,:], scores_te])
    y = np.concatenate([target_for_train[trn_idx], pseudo_label])
    comb = LogisticRegression(solver='lbfgs').fit(X, y)
    oof[val_idx] = comb.predict_proba(scores_tr[val_idx,:])[:,1]
    pred += comb.predict_proba(scores_te)[:,1] / kfold.n_splits
print(f'SCORE (STEP2): {roc_auc_score(target, oof)}')


#################################### STEP3 ####################################
# loacl parameters
target_for_train = np.where(oof>0.5, 1, 0)
pseudo_label = np.where(pred>0.5, 1, 0)
synthetic0_list = deepcopy(samples0_list)
synthetic1_list = deepcopy(samples1_list)
max_clusters = 4

# placeholders
scores_tr = np.zeros((len(train), max_clusters))
scores_te = np.zeros((len(test), max_clusters))

# train and score
train_score(train, target_for_train, magic_tr, test, magic_te,
    use_pseudo=True, pseudo_label=pseudo_label,
    use_synthetic=True, synthetic0_list=synthetic0_list, synthetic1_list=synthetic1_list,
    gen_samples=False)

# combine liklihood scores
oof = np.zeros(len(train))
pred = np.zeros(len(test))
for trn_idx, val_idx in kfold.split(scores_tr, target_for_train):
    X = np.vstack([scores_tr[trn_idx,:], scores_te])
    y = np.concatenate([target_for_train[trn_idx], pseudo_label])
    comb = LogisticRegression(solver='lbfgs').fit(X, y)
    oof[val_idx] = comb.predict_proba(scores_tr[val_idx,:])[:,1]
    pred += comb.predict_proba(scores_te)[:,1] / kfold.n_splits
print(f'SCORE (STEP3): {roc_auc_score(target, oof)}')



# submission
sub =  pd.DataFrame.from_dict({'id': ids_te, 'target': pred})
sub.to_csv('submission.csv', index=False)

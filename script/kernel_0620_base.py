import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

# global parameters
RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)
N_SPLITS = 10; kfold = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
MAX_MAGIC_NO = 5
TRAIN_PATH = '../input/instant-gratification/train.csv'
TEST_PATH = '../input/instant-gratification/test.csv'
PREV_SUBMISSION_PATH = '../input/instant-gratification-sub/submission.csv'

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

# definitions
class GMMBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=1, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        """
        y must be composed of 0 and 1
        """
        self.classes_ = np.unique(y)
        self.gmm0 = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=self.random_state).fit(X[y==0])
        self.gmm1 = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=self.random_state).fit(X[y==1])
        ll = (self.gmm1.score_samples(X) - self.gmm0.score_samples(X)).reshape(-1, 1)
        self.lr_ = LogisticRegression(solver='lbfgs').fit(ll, y)
        return self

    def predict_proba(self, X):
        ll = (self.gmm1.score_samples(X) - self.gmm0.score_samples(X)).reshape(-1, 1)
        return self.lr_.predict_proba(ll)


def train_predict(
    train, target, magic_tr, test, magic_te,
    use_pseudo=False, pseudo_label=None, pseudo_data=None, pseudo_magic=None
    ):

    for i in range(MAX_MAGIC_NO):
        # model
        #clf = GMMBinaryClassifier(n_components=3, random_state=RANDOM_SEED)
        #base_clf = GMMBinaryClassifier(n_components=3, random_state=RANDOM_SEED)
        base_clf = QuadraticDiscriminantAnalysis(reg_param=0.5)
        clf = BaggingClassifier(base_estimator=base_clf, n_estimators=10)
        # data
        train_i = train[magic_tr==i][:,infomative_cols[i]]
        target_i = target[magic_tr==i]
        test_i = test[magic_te==i][:,infomative_cols[i]]
        if use_pseudo:
            pseudo_i = pseudo_data[pseudo_magic==i][:,infomative_cols[i]]
            pseudo_label_i = pseudo_label[pseudo_magic==i]
            pseudo0_i = pseudo_i[pseudo_label_i==0]
            pseudo1_i = pseudo_i[pseudo_label_i==1]
        # placeholders
        oof_i = np.zeros(len(train_i))
        pred_i = np.zeros(len(test_i))

        for j, (trn_idx, val_idx) in enumerate(kfold.split(train_i, target_i)):
            X = train_i[trn_idx]
            y = target_i[trn_idx]
            if use_pseudo:
                X = np.vstack([X, pseudo0_i, pseudo1_i])
                y = np.concatenate([y, np.zeros(len(pseudo0_i)), np.ones(len(pseudo1_i))])
            clf.fit(X, y)
            oof_i[val_idx] = clf.predict_proba(train_i[val_idx,:])[:,1]
            pred_i += clf.predict_proba(test_i)[:,1] / kfold.n_splits
        oof[magic_tr==i] = oof_i
        pred[magic_te==i] = pred_i


#################################### STEP1 ####################################
# placeholders
oof = np.zeros(len(train))
pred = np.zeros(len(test))

# prediction
train_predict(train, target, magic_tr, test, magic_te)

print(f'SCORE (STEP1): {roc_auc_score(target, oof)}')


#################################### STEP2 ####################################
# pseudo targets
target_for_train = np.where(oof>0.5, 1, 0)
pseudo_label = np.where(pred>0.5, 1, 0)

# placeholders
oof = np.zeros(len(train))
pred = np.zeros(len(test))

# prediction
train_predict(train, target_for_train, magic_tr, test, magic_te,
    use_pseudo=True, pseudo_label=pseudo_label, pseudo_data=test, pseudo_magic=magic_te)

print(f'SCORE (STEP2): {roc_auc_score(target, oof)}')



# submission
sub =  pd.DataFrame.from_dict({'id': ids_te, 'target': pred})
sub.to_csv('submission.csv', index=False)

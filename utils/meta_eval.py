from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, pearsonr, kendalltau

def compute_cohen_kappa(annotations, predictions):
    return cohen_kappa_score(annotations, predictions)

def compute_spearman(annotations, predictions):
    return spearmanr(annotations, predictions).correlation

def compute_pearson(annotations, predictions):
    return pearsonr(annotations, predictions)[0]

def compute_kendalltau(annotations, predictions):
    return kendalltau(annotations, predictions).correlation

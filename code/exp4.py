# Significance test for exp2.py

import numpy as np

from metrics import generate_mean_ranking, spearman_footrule
from datasets import load_sancscreen
from parameters import sancscreen_config
from scipy.stats import wilcoxon

from exp2 import run_random_forest, run_neural_network


def spearman(attr, gt_attributions):
    _m_attr = generate_mean_ranking(attr)
    _m_gt = generate_mean_ranking(gt_attributions)
    return spearman_footrule(_m_attr, _m_gt)

if __name__ == "__main__":
    # Load pretrained model, dataset and configs
    d = load_sancscreen()
    c = sancscreen_config
    rng = np.random.RandomState(123)
    feature_names = ["Indicator_value",	"Number_abnormalities",	"hidden_feature_3", "City_hit",	"Account_hit", "Name_hit", "Other_hit",	"hidden_feature_1",	"Country_hit",	"Address_hit",	"Sanction_hit",	"hidden_feature_2",	"Bank_hits", "AccountOwner_hit", "Text_hits", "Specific_Country_hit", "Unique_hits", "Different_hits", "Number_countries"]

    X_train = d.c_train_x.numpy()
    y_train = d.c_train_y.numpy()
    X_test = d.c_test_x.numpy()
    y_test = d.c_test_y.numpy()

    limit_to = -1
    X = d.e_x.numpy()
    y = d.e_y.numpy()
    gt_attributions = d.annot.numpy()

    perm = rng.permutation(np.arange(len(X)))[:limit_to]
    X = X[perm]
    y = y[perm]
    gt_attributions = gt_attributions[perm]

    rf_labels, rf_attributions = run_random_forest(X_train, y_train, X_test, y_test, X, y, gt_attributions, feature_names)
    nn_labels, nn_attributions = run_neural_network(X_train, y_train, X_test, y_test, X, y, gt_attributions, feature_names, c)

    print("RF")
    i = 0
    while i < len(rf_labels):
        j = i+1
        while j < len(rf_labels):
            a_label = rf_labels[i]
            b_label = rf_labels[j]
            a_attr = rf_attributions[i]
            b_attr = rf_attributions[j]

            footrules_a = spearman(a_attr, gt_attributions)
            footrules_b = spearman(b_attr, gt_attributions)
            footrules = footrules_a - footrules_b

            w, p = wilcoxon(footrules)
            if p >= 0.01:
                print("Not significant", a_label, b_label, w, p)

            j += 1
        i += 1

    print()

    print("NN")
    i = 0
    while i < len(nn_labels):
        j = i+1
        while j < len(nn_labels):
            a_label = nn_labels[i]
            b_label = nn_labels[j]
            a_attr = nn_attributions[i]
            b_attr = nn_attributions[j]

            footrules_a = spearman(a_attr, gt_attributions)
            footrules_b = spearman(b_attr, gt_attributions)
            footrules = footrules_a - footrules_b

            w, p = wilcoxon(footrules)
            if p >= 0.01:
                print("Not significant", a_label, b_label, w, p)

            j += 1
        i += 1


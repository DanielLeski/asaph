"""
Copyright 2019 Ronald J. Nowling

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import argparse
import os
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from scipy.stats import chi2
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

def likelihood_ratio_test(features_alternate, labels, lr_model, set_intercept=True, g_scaling_factor=1.0):
    if isinstance(features_alternate, tuple) and len(features_alternate) == 2:
        training_features, testing_features = features_alternate
        training_labels, testing_labels = labels
    else:
        training_features = features_alternate
        testing_features = features_alternate
        training_labels = labels
        testing_labels = labels

    n_training_samples = training_features.shape[0]
    n_testing_samples = testing_features.shape[0]
    n_iter = estimate_lr_iter(n_testing_samples)

    # null model
    null_lr = SGDClassifier(loss = "log",
                            fit_intercept = False,
                            n_iter = n_iter)
    null_training_X = np.ones((n_training_samples, 1))
    null_testing_X = np.ones((n_testing_samples, 1))
    null_lr.fit(null_training_X,
                training_labels)
    null_prob = null_lr.predict_proba(null_testing_X)

    intercept_init = None
    if set_intercept:
        intercept_init = null_lr.coef_[:, 0]

    lr_model.fit(training_features,
                 training_labels,
                 intercept_init = intercept_init)
    alt_prob = lr_model.predict_proba(testing_features)
        
    alt_log_likelihood = -log_loss(testing_labels,
                                   alt_prob,
                                   normalize=False)
    null_log_likelihood = -log_loss(testing_labels,
                                    null_prob,
                                    normalize=False)

    G = g_scaling_factor * 2.0 * (alt_log_likelihood - null_log_likelihood)
    
    # both models have intercepts so the intercepts cancel out
    df = training_features.shape[1]
    p_value = chi2.sf(G, df)

    return p_value

def estimate_lr_iter(n_samples):
    return max(20,
               int(np.ceil(100000. / n_samples)))

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--coordinates",
                        type=str,
                        required=True)

    parser.add_argument("--labels",
                        type=str,
                        required=True)

    parser.add_argument("--columns",
                        type=str,
                        nargs="+",
                        required=True)

    parser.add_argument("--sample-id",
                        type=str,
                        required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    if not os.path.exists(args.coordinates):
        print "Coordinates file path is invalid"
        print args.coordinates
        sys.exit(1)

    if not os.path.exists(args.labels):
        print "Labels file path is invalid"
        print args.labels
        sys.exit(1)
        
    coordinates = pd.read_csv(args.coordinates,
                              delimiter="\t")

    del coordinates["population_index"]
    del coordinates["population_name"]

    n_coordinates = len(coordinates.columns) - 1

    labels = pd.read_csv(args.labels,
                         delimiter="\t")

    joined = coordinates.merge(labels,
                               left_on="sample",
                               right_on=args.sample_id)

    print len(joined), "records survived join"

    # we set the intercept to the class ratios in the lr test function
    lr = SGDClassifier(penalty="l2",
                       loss="log",
                       n_iter = 1000,
                       fit_intercept=True)
    
    for col in args.columns:
        encoder = LabelEncoder()
        class_labels = encoder.fit_transform(joined[col])

        for i in xrange(n_coordinates):
            features = coordinates[str(i + 1)].values.reshape(-1, 1)

            p_value = likelihood_ratio_test(features,
                                            class_labels,
                                            lr,
                                            set_intercept=False)

            lr.fit(features, class_labels)
            pred_labels = lr.predict(features)
            acc = 100. * accuracy_score(class_labels,
                                        pred_labels)

            cm = confusion_matrix(class_labels,
                                  pred_labels)

            print col, encoder.classes_
            print (i+1), p_value, acc
            print cm
            print
        print


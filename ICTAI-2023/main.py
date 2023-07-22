import numpy as np
import pandas as pd
import random
import torch

from Dataset import ImbalancedDataset
from Autoencoder import VAE, SB_VAE
from ctgan.synthesizers.ctgan import CTGAN

# Classification models
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Resampling strategies
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids

from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN

from imblearn.pipeline import make_pipeline

import seaborn as sns
import matplotlib.pyplot as plt


seed = 27
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    datasets = {
                'Rice Classification': ('riceClassification.csv', range(1, 11), 11)
                 }

    d = ImbalancedDataset(seed=seed, dataset=datasets['Rice Classification'])
    dim = d.dimensionality_

    # ctgan = CTGAN(epochs=1)
    # sampled, y = ctgan.fit_resample(d.x_, d.y_)
    # print(sampled)

    # model = VAE(dimensionality=2, latent_dimensionality=3, architecture=[9, 6]).to(device)
    # model.fit(d.x_, d.y_)

    weak_classifier = DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None, random_state=seed)
    clusterer = KMeans(n_clusters=8, init='random', random_state=seed)

    classifiers = {
        # "Logistic Regression": LogisticRegression(max_iter=300, random_state=seed),
        # "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None, random_state=seed),
        # "Random Forest": RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, max_features='sqrt',
        #                                        n_jobs=8, random_state=seed),
        "SVM": SVC(kernel='rbf', C=1, random_state=seed),
        "NeuralNet": MLPClassifier(activation='relu', hidden_layer_sizes=(16, 4), solver='adam', max_iter=300, random_state=seed),
        "AdaBoost": AdaBoostClassifier(base_estimator=weak_classifier, n_estimators=100, algorithm='SAMME.R',
                                       random_state=seed)
    }

    # Under_samplers: A list of tuples of three elements: [ (Method Description, Imbalance Ratio, Algorithm Object) ]
    ratios = [1.0]

    under_samplers = []

    lst = [("RUS-" + str(int(b_ratio * 100)),
            RandomUnderSampler(sampling_strategy=b_ratio, replacement=True, random_state=seed)) for b_ratio in ratios]
    under_samplers.extend(lst)

    lst = [("NM1-" + str(int(b_ratio * 100)),
            NearMiss(version=1, sampling_strategy=b_ratio, n_neighbors=5)) for b_ratio in ratios]
    under_samplers.extend(lst)

    lst = [("NM2-" + str(int(b_ratio * 100)),
            NearMiss(version=2, sampling_strategy=b_ratio, n_neighbors=5)) for b_ratio in ratios]
    under_samplers.extend(lst)

    lst = [("NM3-" + str(int(b_ratio * 100)),
            NearMiss(version=3, sampling_strategy=b_ratio, n_neighbors=5)) for b_ratio in ratios]
    under_samplers.extend(lst)

    lst = [("CLUS-" + str(int(b_ratio * 100)),
            ClusterCentroids(estimator=clusterer, sampling_strategy=b_ratio, random_state=seed)) for b_ratio in ratios]
    under_samplers.extend(lst)
    # print(under_samplers)

    # Over_samplers: A list of tuples of three elements: [ (Method Description, Imbalance Ratio, Algorithm Object) ]
    over_samplers = []

    lst = [("ROS-" + str(int(b_ratio * 100)),
            RandomOverSampler(sampling_strategy=b_ratio, random_state=seed)) for b_ratio in ratios]
    over_samplers.extend(lst)

    lst = [("SMOTE-" + str(int(b_ratio * 100)),
            SMOTE(sampling_strategy=b_ratio, random_state=seed)) for b_ratio in ratios]
    over_samplers.extend(lst)

    lst = [("BORSMOTE-" + str(int(b_ratio * 100)),
            BorderlineSMOTE(sampling_strategy=b_ratio, random_state=seed)) for b_ratio in ratios]
    over_samplers.extend(lst)

    lst = [
        ("SVMSMOTE-" + str(int(b_ratio * 100)),
         SVMSMOTE(sampling_strategy=b_ratio, random_state=seed)) for b_ratio in ratios]
    over_samplers.extend(lst)

    lst = [("CLUSMOTE-" + str(int(b_ratio * 100)),
            KMeansSMOTE(sampling_strategy=b_ratio, cluster_balance_threshold=0.01, random_state=seed))
           for b_ratio in ratios]
    over_samplers.extend(lst)

    lst = [("ADASYN-" + str(int(b_ratio * 100)),
            ADASYN(sampling_strategy=b_ratio, random_state=seed)) for b_ratio in ratios]
    over_samplers.extend(lst)

    lst = [("VAE-" + str(int(b_ratio * 100)),
           VAE(sampling_strategy=b_ratio,
               dimensionality=dim, latent_dimensionality=3, architecture=[12, 8]).to(device))
           for b_ratio in ratios]
    over_samplers.extend(lst)

    lst = [("SB-VAE-" + str(int(b_ratio * 100)),
           SB_VAE(sampling_strategy=b_ratio, radius=0.3,
                  dimensionality=dim, latent_dimensionality=4, architecture=[6, 4]).to(device))
           for b_ratio in ratios]
    over_samplers.extend(lst)


    # lst = [("CTGAN-" + str(int(b_ratio * 100)),
    #       CTGAN(epochs=10, generator_dim=(16, 4), discriminator_dim=(16, 4), sampling_strategy=b_ratio))
    #       for b_ratio in ratios]
    # over_samplers.extend(lst)

    results_list = []

    #pipe_line = Pipeline([
    #    ('Standard Scaler', StandardScaler()),
    #    ('OverSampler', over_samplers[0][1]),
    #    ('Classifier', classifiers['SVM'])])

    #pipe_line = make_pipeline(StandardScaler(), over_samplers[0][1], classifiers['SVM'])
    #print(pipe_line)
    #d.balance(pipe_line, results_list, 'svm', 'ros')
    #exit()

    # Over-sampling Pipelines
    for clf in classifiers:
        for ov_t in over_samplers:
            print("Testing", clf, "with", ov_t[0])
            scaler = StandardScaler()
            #pipe_line = Pipeline([('Standard Scaler', scaler),  ('OverSampler', ov_t[1]), ('Classifier', classifiers[clf])])
            pipe_line = make_pipeline(MinMaxScaler(), ov_t[1], classifiers[clf])

            d.balance(pipe_line, results_list, clf, ov_t[0])

    df_results = pd.DataFrame(
        results_list,
        columns=['Classifier', 'Resampler', 'Accuracy', 'BalancedAccuracy', 'Prec', 'Rec', 'F1']).\
        sort_values(['Classifier'], ascending=[True])

    df_results.to_csv('oversampling_results.csv')

    fig = plt.figure(figsize=(18, 10))

    plt.subplot(3, 1, 1)
    ax = sns.barplot(data=df_results, x="Classifier", y="Accuracy", hue="Resampler", edgecolor='white')
    plt.legend(bbox_to_anchor=(1, 1.07), loc='right', borderaxespad=0, ncol=9)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', fontsize=6)

    plt.subplot(3, 1, 2)
    ax = sns.barplot(data=df_results, x="Classifier", y="BalancedAccuracy", hue="Resampler", edgecolor='white')
    plt.legend(bbox_to_anchor=(1, 1.07), loc='right', borderaxespad=0, ncol=9)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', fontsize=6)

    plt.subplot(3, 1, 3)
    ax = sns.barplot(data=df_results, x="Classifier", y="F1", hue="Resampler", edgecolor='white')
    plt.legend(bbox_to_anchor=(1, 1.07), loc='right', borderaxespad=0, ncol=9)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', fontsize=6)

    plt.subplots_adjust(hspace=0.35)
    plt.show()

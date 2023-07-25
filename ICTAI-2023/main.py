import numpy as np
import pandas as pd
import random
import torch
import gc

from Dataset import ImbalancedDataset
from Autoencoder import VAE, SB_VAE
from ctgan.synthesizers.ctgan import CTGAN, SB_GAN

# Classification models
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler

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

import warnings
warnings.filterwarnings("ignore")


seed = 0
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

torch_state = torch.random.get_rng_state()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def memory_stats():
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_cached()/1024**2)


if __name__ == '__main__':
    datasets = {'Synthetic': None,
                'RiceClassification': ('riceClassification.csv', range(1, 11), 11),
                'Diabetes': ('diabetes.csv', range(0, 8), 8),
                'Surgical': ('Surgical-deepnet.csv', range(0, 24), 24),
                'SmokeDetection': ('smoke_detection_iot.csv', range(2, 15), 15),
                'AndroidPermissions': ('AndroidPermissions.csv', range(0, 86), 86),
                'CreditCardDefault': ('creditcarddefault.csv', range(1, 24), 24)
                }

    KEY = 'CreditCardDefault'
    dataset = datasets[KEY]

    d = ImbalancedDataset(seed=seed, dataset=dataset)
    dim = d.dimensionality_

    weak_classifier = DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None, random_state=seed)
    clusterer = KMeans(n_clusters=8, init='random', random_state=seed)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=seed),
        # "Decision Tree": DecisionTreeClassifier(
        #    criterion='gini', max_depth=None, max_features=None, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, criterion='gini', max_depth=None, max_features='sqrt', n_jobs=1, random_state=seed),
        "SVM": SVC(kernel='rbf', C=1, random_state=seed),
        "NeuralNet": MLPClassifier(
           activation='relu', hidden_layer_sizes=(12, 4), solver='adam', max_iter=300, random_state=seed),
        # "AdaBoost": AdaBoostClassifier(
        #    base_estimator=weak_classifier, n_estimators=50, algorithm='SAMME.R', random_state=seed)
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
    lst = [("ROS-" + str(int(b_ratio * 100)),
            RandomOverSampler(sampling_strategy=b_ratio, random_state=seed)) for b_ratio in ratios]

    # Over-sampling Pipelines
    b_ratio = 1.0
    RAD = 1.0
    results_list = []

    for clf in classifiers:
        generative_models = [
            ("ROS", RandomOverSampler(sampling_strategy=b_ratio, random_state=seed)),
            ("SMOTE", SMOTE(sampling_strategy=b_ratio, random_state=seed)),
            ("BRD-SMOTE", BorderlineSMOTE(sampling_strategy=b_ratio, random_state=seed)),
            ("SVM-SMOTE", SVMSMOTE(sampling_strategy=b_ratio, random_state=seed)),
            ("KMN-SMOTE", KMeansSMOTE(sampling_strategy=b_ratio, cluster_balance_threshold=0.01, random_state=seed)),
            ("ADASYN", ADASYN(sampling_strategy=b_ratio, random_state=seed)),
            ("VAE", VAE(torch_state, sampling_strategy=b_ratio, dimensionality=dim, latent_dimensionality=3,
                        architecture=[9, 6]).to(device)),
            ("SB-VAE", SB_VAE(torch_state, sampling_strategy=b_ratio, dimensionality=dim, latent_dimensionality=3,
                              architecture=[9, 6], radius=RAD).to(device)),
            ("GAN", CTGAN(torch_state, sampling_strategy=b_ratio, epochs=10, embedding_dim=8,
                          generator_dim=(32, 16), discriminator_dim=(32, 16), pac=1)),
            ("SB-GAN", SB_GAN(torch_state, sampling_strategy=b_ratio, epochs=10, embedding_dim=8,
                              generator_dim=(32, 16), discriminator_dim=(32, 16), pac=1, radius=RAD))
        ]

        order = 0
        for mdl in generative_models:
            order = order + 1
            print("Testing", clf, "with", mdl[0])

            scaler = StandardScaler()
            pipe_line = make_pipeline(scaler, mdl[1], classifiers[clf])
            d.balance(pipe_line, results_list, clf, mdl[0], order)

            torch.cuda.empty_cache()
            del pipe_line
            del mdl
            gc.collect()
            torch.cuda.empty_cache()

    df_results = pd.DataFrame(
        results_list,
        columns=['Classifier', 'Resampler', 'Order',
                 'Accuracy_mean', 'Accuracy_std',
                 'BalAccuracy_mean', 'BalAccuracy_std',
                 'AUC_mean', 'AUC_std',
                 'F1_mean', 'F1_std'
                 ]).sort_values(['Classifier', 'Order'], ascending=[True, True])

    df_results.to_csv(KEY+'.csv')

    fig = plt.figure(figsize=(18, 10))

    plt.subplot(3, 1, 1)
    ax = sns.barplot(data=df_results, x="Classifier", y="Accuracy_mean", hue="Resampler", edgecolor='white')
    plt.legend(bbox_to_anchor=(1, 1.07), loc='right', borderaxespad=0, ncol=10)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', fontsize=7)

    plt.subplot(3, 1, 2)
    ax = sns.barplot(data=df_results, x="Classifier", y="AUC_mean", hue="Resampler", edgecolor='white')
    plt.legend(bbox_to_anchor=(1, 1.07), loc='right', borderaxespad=0, ncol=10)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', fontsize=7)

    plt.subplot(3, 1, 3)
    ax = sns.barplot(data=df_results, x="Classifier", y="F1_mean", hue="Resampler", edgecolor='white')
    plt.legend(bbox_to_anchor=(1, 1.07), loc='right', borderaxespad=0, ncol=10)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', fontsize=7)

    fig.savefig(KEY+'.pdf', bbox_inches='tight')

    plt.subplots_adjust(hspace=0.35)
    plt.show()

# text processing datasets
import time
import pathlib
import gzip
import json
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


class ImbalancedDataset:
    # Dataset initializer
    def __init__(self, dataset=None, seed=42):
        self.seed_ = seed
        self.dimensionality_ = 0

        if dataset is None:
            self.num_samples_ = 1000
            self.imbalance_ratio_ = [0.95, 0.05]
            self.num_classes_ = len(self.imbalance_ratio_)
            synthetic_dataset = make_classification(
                n_samples=self.num_samples_, n_features=2, n_clusters_per_class=2, flip_y=0,
                n_classes=self.num_classes_, weights=self.imbalance_ratio_, class_sep=0.5,
                n_informative=2, n_redundant=0, n_repeated=0, random_state=seed)

            self.x_ = synthetic_dataset[0]
            self.y_ = synthetic_dataset[1]

        else:
            path = 'C:/Users/Leo/Documents/JupyterNotebooks/Imbalanced Data/datasets/' + dataset[0]
            file_extension = pathlib.Path(path).suffix
            if file_extension == '.csv':
                self.df_ = pd.read_csv(path, encoding='utf-8')
                # self.df_ = pd.read_csv(path, encoding='latin-1', header=None)
            else:
                self.df_ = self.getDF(path)

            self.x_ = self.df_.iloc[:, dataset[1]].to_numpy()
            self.y_ = self.df_.iloc[:, dataset[2]].to_numpy()

            class_encoder = LabelEncoder()
            self.y_ = class_encoder.fit_transform(self.y_)

        self.num_classes_ = len(list(set(self.y_)))

        # Run data preprocessor
        self.dimensionality_ = self.x_.shape[1]
        self.pre_process()

        print("Class Distribution:")
        for k in range(self.num_classes_):
            print("Class", k, ":", len(self.y_[self.y_ == k]), "samples")

    # Apply preprocessing of the input data
    def pre_process(self):
        t0 = time.time()

    # Perform text Processing. This includes two steps: i) Text Vectorization and ii) Dimensionality Reduction.
    def balance(self, balancing_pipeline, results_list, classifier_str, sampler_str):
        cv_results = cross_validate(
            balancing_pipeline, self.x_, self.y_,
            cv=5, scoring=['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1'],
            return_train_score=True, return_estimator=True, n_jobs=8)

        # print(classifier_str, sampler_str, cv_results['test_balanced_accuracy'])
        results_list.append([
            classifier_str, sampler_str,
            cv_results['test_accuracy'].mean(),
            cv_results['test_balanced_accuracy'].mean(),
            cv_results['test_precision'].mean(),
            cv_results['test_recall'].mean(),
            cv_results['test_f1'].mean()
        ])

    # Return the dataframe
    def get_data(self):
        return self.df_

    def getDF(self, path):
        i = 0
        df = {}
        for d in self.parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    def parse(self, path):
        g = gzip.open(path, 'rb')
        for lt in g:
            yield json.loads(lt)

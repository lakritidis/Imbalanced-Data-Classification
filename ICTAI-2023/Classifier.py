import numpy as np
import time

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Evaluation Measures
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score


def plot_decision_regions_2d(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 'o', 'o', 'o', 's', 'v', '^')
    colors = ('#1f77b4', '#ff7f0e', 'red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    x2_min, x2_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    # meshgrid: Return coordinate matrices from coordinate vectors. More specifically, we make N-D coordinate arrays
    # for vectorized evaluations of N-D scalar/vector fields over N-D grids, given one-dimensional coordinate arrays.
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # ravel: Return a contiguous flattened array.
    # T: the transpose matrix
    x_test_in = np.array([xx1.ravel(), xx2.ravel()]).T
    # print(X_test)

    z = classifier.predict(x_test_in)
    z = z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, z, alpha=0.15, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl,
                    edgecolor='white')


def train_test_model(train_x, test_x, train_y, test_y, mdl, res, desc):
    t0 = time.time()
    print("Training", mdl + "...\t", end="", flush=True)

    clf = models[mdl]
    clf.fit(train_x, train_y)

    print("  (%5.3f sec). \t" % (time.time() - t0), end="", flush=True)

    y_predicted = clf.predict(test_x)

    acc = accuracy_score(test_y, y_predicted)
    bal_acc = balanced_accuracy_score(test_y, y_predicted)
    precision = precision_score(test_y, y_predicted)
    recall = recall_score(test_y, y_predicted)
    f1 = f1_score(test_y, y_predicted)

    print("Accuracy=%5.4f" % acc, "\tBalanced Accuracy=%5.4f" % bal_acc,
          "\tPrecision=%5.4f" % precision, "\tRecall=%5.4f" % recall, "\tF1=%5.4f" % f1, flush=True)

    res.append([mdl, desc, acc, bal_acc, precision, recall])

    x_stacked = np.vstack((train_x, test_x))
    y_stacked = np.hstack((train_y, test_y))

    plot_decision_regions_2d(x_stacked, y_stacked, clf)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title(mdl + " (Accuracy: " + str(round(acc, 3)) + ", AUC: " + str(round(bal_acc, 3)) + ")")
    plt.legend(loc='upper left')

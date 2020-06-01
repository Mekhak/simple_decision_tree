import numpy as np
import matplotlib.pyplot as plt

from decision_tree import DecisionTree


def generate_data():
    data0 = np.vstack((np.random.multivariate_normal(mean=[1, 4],
                                                     cov=[[7, 1], [1, 7]],
                                                     size=50),
                       np.random.multivariate_normal(mean=[14, -14],
                                                     cov=[[5, 1], [1, 5]],
                                                     size=50)
                       ))
    labels0 = np.zeros(100, dtype=int)

    data1 = np.random.multivariate_normal(mean=[3, -3], cov=[[8, 2], [2, 8]],
                                          size=100)
    labels1 = np.ones(100, dtype=int)

    data2 = np.random.multivariate_normal(mean=[15, 5], cov=[[8, 2], [2, 8]],
                                          size=100)
    labels2 = 2 * np.ones(100, dtype=int)

    data = np.vstack((data0, data1, data2))
    labels = np.concatenate((labels0, labels1, labels2))
    shuffle_array = np.arange(len(labels))
    np.random.shuffle(shuffle_array)
    data = data[shuffle_array]
    labels = labels[shuffle_array]
    return data, labels


def plot(data, labels):
    decision_tree = DecisionTree()
    decision_tree.fit(data, labels)

    # plt.figure(figsize=(16, 9))

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min)/500),
                         np.arange(y_min, y_max, (y_max - y_min)/500))

    space = np.vstack((xx.ravel(), yy.ravel())).T
    predictions = []
    for point in space:
        pred = decision_tree.predict(point)
        predictions.append(pred)
    predictions = np.array(predictions)
    plt.figure(figsize=(16, 9))
    plt.contourf(xx, yy, predictions.reshape(xx.shape), cmap=plt.cm.RdYlBu);
    plt.scatter(data.T[0], data.T[1], c=labels);


if __name__ == '__main__':
    data, labels = generate_data()
    plot(data, labels)
    plt.show()

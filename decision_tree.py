import numpy as np


class DecisionNode:
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 column: int = None,
                 value: float = None,
                 false_branch = None,
                 true_branch = None,
                 is_leaf: bool = False):
        """
        Building block of the decision tree.

        :param data: numpy 2d array data can for example be
         np.array([[1, 2], [2, 6], [1, 7]])
         where [1, 2], [2, 6], and [1, 7] represent each data point
        :param labels: numpy 1d array
         labels indicate which class each point belongs to
        :param column: the index of feature by which data is splitted
        :param value: column's splitting value
        :param true_branch(false_branch): child decision node
        true_branch(false_branch) is DecisionNode instance that contains data
        that satisfies(doesn't satisfy) the filter condition.
        :param is_leaf: is true when node has no child

        """
        self.data = data
        self.labels = labels
        self.column = column
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.is_leaf = is_leaf


class DecisionTree:

    def __init__(self,
                 max_tree_depth=4,
                 criterion="gini",
                 task="classification"):
        self.tree = None
        self.max_depth = max_tree_depth
        self.task = task

        if criterion == "entropy":
            self.criterion = self._entropy
        elif criterion == "square_loss":
            self.criterion = self._square_loss
        elif criterion == "gini":
            self.criterion = self._gini
        else:
            raise RuntimeError(f"Unknown criterion: '{criterion}'")

    @staticmethod
    def _gini(labels: np.ndarray) -> float:
        """
        Gini criterion for classification tasks.

        """
        _, counts_elements = np.unique(labels, return_counts=True)

        gini = 1 - sum((counts_elements / len(labels)) ** 2)

        return gini

    @staticmethod
    def _entropy(labels: np.ndarray) -> float:
        """
        Entropy criterion for classification tasks.

        """

        _, counts_elements = np.unique(labels, return_counts=True)

        p = counts_elements / len(labels)

        entropy = - sum(p * np.log(p))

        return entropy


    @staticmethod
    def _square_loss(labels: np.ndarray) -> float:
        """
        Square loss criterion for regression tasks.

        """

        y_hat = np.mean(labels)

        mse = (1 / len(labels)) * sum((labels - y_hat)**2)

        return mse

    def _iterate(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 current_depth=0) -> DecisionNode:
        """
        This method creates the whole decision tree, by recursively iterating
         through nodes.
        It returns the first node (DecisionNode object) of the decision tree,
         with it's child nodes, and child nodes' children, ect.
        """

        if len(labels) == 1:
            # return a node is_leaf=True
            return DecisionNode(data = data, labels = labels, is_leaf=True)

        impurity = self.criterion(labels)
        best_column, best_value = None, None

        for column, column_values in enumerate(data.T):
            for split_value in np.arange(
                    min(column_values), max(column_values),
                    (max(column_values) - min(column_values)) / 50):

                false_labels = labels[column_values <  split_value]
                true_labels  = labels[column_values >= split_value]

                false_impurity = self.criterion(false_labels)
                true_impurity = self.criterion(true_labels)

                final_impurity = (len(false_labels) / len(labels)) * false_impurity + \
                                 (len(true_labels)  / len(labels)) * true_impurity


                if final_impurity < impurity:
                    impurity = final_impurity
                    best_value = split_value
                    best_column = column


        if best_column is None or current_depth == self.max_depth:
            return DecisionNode(data, labels, is_leaf=True)
        else:
            # return DecisionNode with true(false)_branch=self._iterate(...)
            false_data = data[(data[:, [best_column]]).flatten() <  best_value]
            true_data  = data[(data[:, [best_column]]).flatten() >= best_value]

            false_labels = labels[(data[:, [best_column]]).flatten() <  best_value]
            true_labels =  labels[(data[:, [best_column]]).flatten() >= best_value]

            return DecisionNode(data = data, labels = labels,
                                column = best_column, value = best_value,
                                false_branch = self._iterate(false_data, false_labels, current_depth + 1),
                                true_branch =  self._iterate(true_data, true_labels, current_depth + 1))


            # raise NotImplementedError()

    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.tree = self._iterate(data, labels)

    def predict(self, point: np.ndarray) -> float or int:
        """
        This method iterates nodes starting with the first node i. e.
        self.tree. Returns predicted label of a given point (example [2.5, 6],
        where 2.5 and 6 are points features).

        """
        node = self.tree

        while True:
            if node.is_leaf:
                if self.task == "classification":
                    # predict and return the label for classification task
                    return np.bincount(node.labels).argmax()
                else:
                    # predict and return the label for regression task
                    return np.mean(node.labels)

            if point[node.column] >= node.value:
                node = node.true_branch
            else:
                node = node.false_branch

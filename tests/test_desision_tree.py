import unittest
import numpy as np

from decision_tree import DecisionTree, DecisionNode


class TestDecisionTree(unittest.TestCase):

    def test_gini(self):
        decision_tree = DecisionTree()
        labels = np.array([0, 1])
        self.assertAlmostEqual(decision_tree._gini(labels), 0.5)

    def test_entropy(self):
        decision_tree = DecisionTree()
        labels = np.array([1, 1])
        self.assertAlmostEqual(decision_tree._entropy(labels), 0)

    def test_square_loss(self):
        decision_tree = DecisionTree()
        labels = np.array([0, 0])
        self.assertAlmostEqual(decision_tree._square_loss(labels), 0)

    def test_iterate(self):
        decision_tree = DecisionTree()
        data = np.array([[1], [2], [3]])
        labels = np.array([0, 0, 1])
        node = decision_tree._iterate(data, labels)
        self.assertTrue(isinstance(node, DecisionNode))

    def test_fit_predict_classification(self):
        decision_tree = DecisionTree()
        data = np.array([[1], [2], [3]])
        labels = np.array([0, 0, 1])
        decision_tree.fit(data, labels)
        pred = decision_tree.predict([3.5])
        self.assertIn(pred, labels)

    def test_fit_predict_regression(self):
        decision_tree = DecisionTree(task="regression")
        data = np.array([[1], [2], [3]])
        labels = [0.5, 0.25, 1.5]
        decision_tree.fit(data, np.array(labels))
        pred = decision_tree.predict([2.5])
        self.assertTrue(isinstance(pred, float))

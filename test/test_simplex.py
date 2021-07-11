from unittest import TestCase

import numpy as np
from numpy.linalg import LinAlgError

from simplex import LinearProgram, get_vertex, edge_transition, get_lambda, get_mu_v


"""
Questi test si basano sull'esempio 11.7 a pagina 203, prima di cambiare valori è meglio vedere bene che matrici ho e 
assicurarsi che siano giuste e ritestare con altre matrici.
"""

class Test(TestCase):
    def setUp(self) -> None:
        A = np.array([[1, 1, 1, 0], [-4, 2, 0, 1]])
        b = np.array([[9], [2]])
        x = np.empty([4, 1])
        c = np.array([[3], [-1], [0], [0]])
        self.LP = LinearProgram(A, b, c)

    def test_get_vertex(self):
        vertex = get_vertex(np.array([2, 3]), LP=self.LP)
        np.testing.assert_array_almost_equal(vertex, np.array([[9], [2]]))

    def test_get_vertex_with_ordering_important(self):
        vertex = get_vertex(np.array([3, 2]), LP=self.LP)
        np.testing.assert_array_almost_equal(vertex, np.array([[9], [2]]))

    def test_get_vertex_raises_error_when_singular(self):
        A = np.array([[1, 1, 1, 0], [2, 2, 0, 1]])
        b = np.array([[9], [2]])
        x = np.empty([4, 1])
        c = np.array([[3], [-1], [0], [0]])
        LP = LinearProgram(A, b, c)
        self.assertRaises(LinAlgError, get_vertex, np.array([0, 1]), LP=LP)

    def test_edge_transition_correct_xq_value(self):
        B = np.array([2, 3])
        mu_v = np.array([3, -1])
        q = 1
        p, xq = edge_transition(self.LP, B, q)
        self.assertEqual(1, xq)

    def test_edge_transition_correct_exit_column_pivoting(self):
        B = np.array([2, 3])
        mu_v = np.array([3, -1])
        q = 1
        p, xq = edge_transition(self.LP, B, q)
        self.assertEqual(1, p)

    def test_get_lambda_using_example_11_7(self):
        lambda_vect = get_lambda(np.array([[1, 0], [0, 1]]), np.array([[0], [0]]))
        np.testing.assert_array_almost_equal(lambda_vect, np.array([[0], [0]]))

    def test_get_mu_v(self):
        cv = np.array([[3], [-1]])
        Ab = np.array([[1, 0], [0, 1]])
        Av = np.array([[1, 1], [-4, 2]])
        cb = np.array([[0], [0]])
        multiplier_non_neg = get_mu_v(cv, Ab, Av, cb)
        np.testing.assert_array_almost_equal(multiplier_non_neg, np.array([[3], [-1]]))


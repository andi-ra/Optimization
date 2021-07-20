import logging
import unittest
from unittest import TestCase
import numpy as np
from simplex import ConstraintMatrix, LinearProgram, CostVector

logging.basicConfig(level=logging.WARNING)


class TestConstraintMatrix(TestCase):
    def setUp(self) -> None:
        # INIZIALIZZAZIONE E RISULTATI DEL PROBLEMA DI TEST
        self.A = np.array([[1, 2, 3, 1, -3, 1, 0, 0], [2, -1, 2, 2, 1, 0, 1, 0], [-3, 2, 1, -1, 2, 0, 0, 1]])
        self.B = np.array([5, 6, 7])
        self.b = np.array([[9], [10], [11]])
        self.c = CostVector(np.array([[4], [3], [1], [7], [6], [0], [0], [0]]), self.B)
        self.constraint_matrix = ConstraintMatrix(self.A, self.B)
        self.N = np.array([[1, 2, 3, 1, -3], [2, -1, 2, 2, 1], [-3, 2, 1, -1, 2]])
        self.LP = LinearProgram(self.constraint_matrix, self.b, self.c)
        self.non_base_cost = np.array([[4], [3], [1], [7], [6]])
        self.non_base_cost_index = np.array([0, 1, 2, 3, 4])
        self.reduced_non_base_costs = np.array([-4, -3, -1, -7, -6])
        self.entering_column_index = 3  # cioè x4
        self.entering_column = np.array([[1], [2], [-1]])  # cioè x4
        self.exiting_column_index_non_base_space = 1  # cioè s2, la seconda colonna nella matrice di base
        self.leaving_variable_reduced_cost = -7  # cioè il c4 ridotto, quello che va nell'obiettivo al pivoting
        self.leaving_column = np.array([0, 1, 0])  # Questa è la colonna associata ad s2 nella base matrix
        self.pivot_element = np.array([2])
        self.pivot_element_after_one_iter = np.array([1])
        self.previous_pivot_element = np.array([2])
        self.first_iteration_base_matrix = np.array([[1, -0.5, 0], [0, 0.5, 0], [0, 0.5, 1]])
        self.second_iteration_base_matrix = np.array([[0.4, -0.2, 0], [0.2, 0.4, 0], [-0.6, 0.8, 1]])
        self.b_first_iteration = np.array([[4], [5], [16]])
        self.b_second_iteration = np.array([[1.6], [5.8], [13.6]])
        self.second_iteration_reduced_costs = np.array([3, -6.5, 6, 0, -2.5, 0, 3.5, 0])
        self.second_iteration_entering_column = np.array([[2.5], [-0.5], [1.5]])

    def test_non_base_indexes_return_correct_matrix(self):
        np.testing.assert_array_almost_equal(self.N, self.constraint_matrix.non_base_matrix)

    def test_raise_runtime_error_for_create_empty_constr_matrix(self):
        self.assertRaises(RuntimeError, ConstraintMatrix, np.empty([0, 0]), np.empty([0, 0]))

    def test_raise_runtime_error_for_create_empty_base_partition(self):
        self.assertRaises(RuntimeError, ConstraintMatrix, np.empty([0, 0]), np.empty([0, 0]))

    def test_objective_value(self):
        np.testing.assert_array_almost_equal(0, self.LP.objective_value)

    def test_get_dual_variable_in_single_step(self):
        np.testing.assert_array_almost_equal(self.LP.dual_variable, np.zeros(self.b.shape))

    def test_exit_column_index(self):
        # Attenzione la colonna 4 del libro è la 3 di python!!
        self.assertEqual(3, self.LP.entering_column_index)

    def test_reduced_non_base_cost(self):
        # Questo testa il vettore dei costi ridotti delle variabili non in base
        np.testing.assert_array_almost_equal(self.LP.reduced_cost_non_base,
                                             self.reduced_non_base_costs.reshape(self.LP.reduced_cost_non_base.shape))

    def test_non_base_cost(self):
        np.testing.assert_array_almost_equal(self.LP.cost_vector.non_base_cost, self.non_base_cost)

    def test_non_base_index(self):
        np.testing.assert_array_almost_equal(self.LP.cost_vector.non_base_indexes, self.non_base_cost_index)

    def test_entering_column_index(self):
        self.assertEqual(self.entering_column_index, self.LP.entering_column_index)

    def test_entering_column(self):
        np.testing.assert_array_almost_equal(self.entering_column, self.LP.entering_column)

    def test_leaving_column_index(self):
        self.assertEqual(self.exiting_column_index_non_base_space, self.LP.exit_variable_index)

    def test_exit_variable_reduced_cost(self):
        self.assertEqual(self.leaving_variable_reduced_cost, self.LP.entering_variable_reduced_cost)

    def test_exit_variable_column(self):
        np.testing.assert_array_almost_equal(self.leaving_column, self.LP.exit_variable_row)

    def test_pivot_element(self):
        np.testing.assert_array_almost_equal(self.pivot_element, self.LP.current_pivot_element)

    def test_update_base_matrix(self):
        self.LP.update_revised_simplex_matrix_entire_rows()
        np.testing.assert_array_almost_equal(self.first_iteration_base_matrix,
                                             self.LP.constraint_matrix.base_matrix)

    def test_update_base_matrix_two_iterations(self):
        next(self.LP)
        next(self.LP)
        np.testing.assert_array_almost_equal(self.second_iteration_base_matrix, self.LP.constraint_matrix.base_matrix)

    def test_update_revised_simplex_known_terms_second_iteration(self):
        next(self.LP)
        next(self.LP)
        np.testing.assert_array_almost_equal(self.b_second_iteration, self.LP._known_terms)

    def test_update_revised_simplex_reduced_costs(self):
        next(self.LP)
        np.testing.assert_array_almost_equal(
            self.second_iteration_reduced_costs.reshape(self.LP.reduced_cost_total_vector.shape),
            self.LP.reduced_cost_total_vector)

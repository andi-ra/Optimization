# I PROBLEMI DEVONO ESSERE IN FORMA UGUAGLIANZA

import numpy as np


class CostVector:
    """Questa classe rappresenta il vettore dei costi del problema.

    La creazione della classe permette sempre di avere i metodi accessori per le proprietà fondamentali del vettore dei
    costi:
        * Accessori per il vettore dei costi delle variabili in base
        * Accessori per il vettore dei costi delle variabili NON in base
    La seconda declinazione della classe `CostVector` è rappresentare il vettore dei costi ridotti, cioè quel vettore
    dei costi nello spazio completo, :math:`\mathbb{R}^{m+n}` in questo caso uso lo :math:`span<>` di tutte le colonne.
    """

    def __init__(self, cost_vector: np.ndarray, base_columns: np.ndarray):
        if base_columns is None:
            base_columns = np.empty([0])
        self._c = cost_vector
        self._base_indexes = base_columns
        if self._c.size != 0:
            self._indexes = np.array(np.arange(0, self._c.shape[0]))
        else:
            self._indexes = np.empty([0])
        self._non_base_index = np.delete(self._indexes, self._base_indexes)
        self.mu_v = np.empty([0])

    @property
    def base_cost(self):
        return self._c[self._base_indexes]

    @property
    def non_base_cost(self):
        return self._c[self._non_base_index]

    @property
    def non_base_index(self):
        return self._non_base_index


class ConstraintMatrix:
    """Represents the constraint matrix :math:`A`

        Questa classe rappresenta la matrice dei vincoli del problema. È stata aggiornata con metodi accessori per
        garantire la consistenza della matrice durante le varie operazioni. Secondo il motto
        "`We are all consenting adults <https://python-guide-chinese.readthedocs.io/zh_CN/latest/writing/style.html
        #we-are-all-consenting-adults>`_" possono essere modificati ma queste guardie proteggono contro accidentalità.

        .. Caution:: I PROBLEMI DEVONO ESSERE IN FORMA STANDARD. [notazione]_

        .. [notazione] Algorithms for Optimization, Mykel J. Kochenderfer, Tim A. Wheeler
    """

    def __init__(self, constraint_matrix: np.ndarray, base_indexes: np.ndarray) -> None:
        """
        Questo metodo inizializza le seguenti strutture dati:
            * Matrice dei vincoli totale del problema
            * Insieme degli indici che rappresentano le colonne della matrice di base
            * Insieme degli indici che rappresentano le colonne della matrice NON di base
        La cosa particolare è che l'insieme degli indici è fatto tramite un *mutable* dunque bisogna fare attenzione a
        come viene usato, in quanto se inizializzassi nuovamente l'insieme degli indici questo conserva lo stato come
        `qui <https://docs.python-guide.org/writing/gotchas/>`_. Per evitare, metto una guardia sul empty set.

        :param constraint_matrix: Matrice dei vincoli da dare in pasto alla classe ConstraintMatrix
        :param base_indexes: Questa è la partizione iniziale che devi fornire se non vuoi risolvere un prima fase.
        """
        self._A = constraint_matrix
        self._B = base_indexes
        self._indexes = np.empty([0, 0])
        if self._A.size != 0:
            self._indexes = np.array(np.arange(0, self._A.shape[1]))
        else:
            raise RuntimeError("Can't start with empty constraint matrix")
        if self._B.size == 0:
            raise RuntimeError("Solve a 1-st phase problem first!!")
        self._N = np.delete(self._indexes, self._B, axis=0)

    @property
    def non_base_indexes(self):
        return np.delete(self._indexes, self._B, axis=0)

    @property
    def base_matrix(self) -> np.ndarray:
        return self._A[:, self._B]

    @property
    def constraint_matrix(self):
        return self._A

    @constraint_matrix.setter
    def constraint_matrix(self, new_constraint_matrix: np.ndarray):
        if new_constraint_matrix.size != 0:
            self._A = new_constraint_matrix
        else:
            raise ValueError("Cannot update old matrix with an empty one!")

    @property
    def non_base_matrix(self):
        return self._A[:, self._N]

    @property
    def base_indexes(self):
        return self._B

    @base_indexes.setter
    def base_indexes(self, new_base_indexes: np.ndarray = np.empty([0])):
        if new_base_indexes is None:
            new_base_indexes = np.empty([0])
        else:
            self._B = new_base_indexes


class LinearProgram:
    """Classe che rappresenta il problema lineare da ottimizzare"""

    def __init__(self, constr_matrix: ConstraintMatrix, b: np.ndarray, c: CostVector) -> None:
        self.constraint_matrix = constr_matrix
        self.known_terms = b
        self.cost_vector = c
        # self.optimalValue = None
        # self.feasible = True
        # self.bounded = False
        # # self.mu_v = np.empty([0, 0])

    @property
    def solution_vertex(self):
        return np.linalg.solve(self.constraint_matrix.base_matrix, self.known_terms)

    @property
    def objective_value(self):
        return self.dual_variable * self.known_terms

    @property
    def dual_variable(self) -> np.ndarray:
        """
        Funzione usata per il calcolo dei moltiplicatori di Lagrange lambda. Nel caso di problemi lineari so che i
        moltiplicatori di Lagrange corrispondono alle variabili duali. Queste variabili duali le uso per calcolarmi il
        certificato duale a fine ottimizzazione. In pratica ciò che faccio è moltiplicare il vettore dei costi delle
        variabili di base con la matrice dei vincoli delle variabili di base.
        Calcolo delle variabili duali: :math:`c^T B^{-1} = u^T`

        :return: Vettore dei moltiplicatori di Lagrange lambda
        """
        return np.linalg.solve(np.matrix.transpose(self.constraint_matrix.base_matrix), self.cost_vector.base_cost)

    @property
    def entering_column_index(self) -> int:
        exit_column_index = np.where(self.reduced_non_base_cost == np.amin(self.reduced_non_base_cost))
        return int(np.unique(np.asarray(exit_column_index[0])))

    @property
    def entering_column(self) -> np.ndarray:
        return np.reshape(np.linalg.solve(self.constraint_matrix.base_matrix,
                                          self.constraint_matrix.constraint_matrix[:, self.entering_column_index]),
                          self.known_terms.shape)

    @property
    def entering_variable_reduced_cost(self):
        return self.reduced_non_base_cost[self.entering_column_index]

    @property
    def reduced_non_base_cost(self):
        """
        Funzione che restituisce il secondo vettore di moltiplicatori di Lagrange (quello associato ai vincoli di non
        negatività). Questo vettore nella dimostrazione di [WHLR]_ viene scomposto fra costi delle variabili di base
        (che viene posto reduced_non_base_cost zero) e quelli NON di base (che sono oggetto diu questa funzione). In
        pratica nel libro [APPL1]_ si tratta dei costi ridotti (cioè :math:`\mu_v  \in \mathbb{R}^{m+n}` ). Questi
        costi li ispezionerò per controllare l'ottimalità della soluzione corrente, se c'è anche sola una componente
        negativa, questa viola la duale ammissibilità e quindi necessariamente non può essere soluzione ottima.

        .. [WHLR] Wheeler, Algorithm for optimization sezione 11.2.2 .. [APPL1] libro Applied Integer programming
        pag. 234-235 Algoritmo Revised Simplex :param Ab: Matrice di base :param non_base_constraint_matrix: Matrice
        NON di base (colonne associate alle variabili non di base, nel tableau) :return: Vettore associato al
        variabili NON di base dei moltiplicatori associato reduced_non_base_cost vincoli di non negatività

        .. warning::
            DEVI AGGIUSTARE LE DIMENSIONI... PER ORA FUNZIONA MA NON È DETTO CHE IN FUTURO CONTINUI

            ValueError: shapes (3,1) and (3,5) not aligned: 1 (dim 1) != 3 (dim 0)
        """
        reduced_non_base_cost = np.dot(np.transpose(self.dual_variable), self.constraint_matrix.non_base_matrix) - \
                                self.cost_vector.non_base_cost[self.cost_vector.non_base_index]
        return reduced_non_base_cost[:, 0]

    @property
    def pivot_element(self):
        column_pivot = self.entering_column_index
        row_pivot = self.exit_variable_index
        return self.constraint_matrix.constraint_matrix[row_pivot, column_pivot]

    @property
    def exit_variable_index(self) -> int:
        """
        Questa funzione implementa la transizione da un vertice al successivo per la fase di ottimizzazione. Nel metodo
        del simplesso mi devo muovere da un vertice al successivo, qui uso l'euristica del MIN_RATIO. Questa operazione
        nei libri classici è detta [APPL1]_ PIVOTING. Per ogni variabile NON di base j e per ogni colonna a_j della
        matrice dei coefficienti NON di base associata a quel j, questa funzione calcola la colonna nello spazio ridotto
        (quello m+n-dimensionale). Inoltre si calcola il [WHLR]_ :math:`x_q` e cioè quel valore che porta la variabile
        più vicina al suo vincolo a sbatterci contro, cioè una delle variabili andrà a zero, è proprio quella indice q
        con valore associato x_q', in uscita avrò proprio questo.

        ATTENZIONE QUESTA RAGIONA IN TERMINI DI COSTI E VARIBIALI NON DI BASE, DUNQUE NON NELLO SPAZIO RIDOTTO!!
        +-----------------------------------------------+
        |                   Pivot row                   |
        +===============================================+
        | :math:`min[ b/a_ik | a_ik > 0]`               |
        +-----------------------------------------------+
        |cardine :math:`a_ik` divido le righe           |
        |per questo valore.                             |
        +-----------------------------------------------+

        .. [APPL1] libro Applied Integer programming pag. 234-235 Algoritmo Revised Simplex
        .. [WHLR] Wheeler, Algorithm for optimization sezione 11.2.3

        :return: Coppia (indice della variabile in uscita, valore che la azzera)
        """
        ratio = []
        d = self.entering_column
        for index in range(0, len(d)):
            if d[index] > 0:
                ratio.append({'ratio_value': float(self.known_terms[index] / d[index]), 'ratio_index': index})
        ratios = [x['ratio_value'] for x in ratio]
        min_ratio = min(ratios)
        exit_variable_index = [x['ratio_index'] for x in ratio if x['ratio_value'] == min_ratio]
        return int(exit_variable_index[0])

    @property
    def exit_variable_column(self):
        return self.constraint_matrix.base_matrix[self.exit_variable_index]

    # def update_revised_simplex_tableau(self):
    #     pass

# def solve_LP_no_first_base(B: np.array, LP: LinearProgram):
#     A = LP.constraint_matrix
#     b = LP.known_terms
#     c = LP.cost_vector
#     m, n = A.shape
#     z = np.ones(m)
#     lista = [+1 if b[j] >= 0 else -1 for j in b if b[j] >= 0]
#     a = np.array(lista)
#     Z = np.diag(a)
#     A_concat = np.concatenate(A, Z, 1)
#     b_new = b
#     c_new = np.concatenate(np.zeros(n), z, 0)
#     LP_init = LinearProgram(A_concat, b_new, c_new)
#     B = np.arange(start=1, stop=m + 1, step=1) + n
#
#     for i in B:
#         if B[i] > n:
#             raise ValueError("Infeasible")
#
#     solve_LP_no_first_base(B, LP_init)

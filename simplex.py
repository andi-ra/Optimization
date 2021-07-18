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

    @property
    def base_cost(self):
        return self._c[self._base_indexes]

    @property
    def non_base_cost(self):
        return self._c[self._non_base_index]

    @non_base_cost.setter
    def non_base_cost(self, value: np.ndarray):
        # TODO: controllare che tutto torni e ritornare eccezioni corrette
        self._c[self._non_base_index] = value

    @property
    def non_base_indexes(self):
        return self._non_base_index

    @non_base_indexes.setter
    def non_base_indexes(self, value: np.ndarray):
        self._non_base_index = value

    @property
    def base_indexes(self):
        return self._base_indexes

    @base_indexes.setter
    def base_indexes(self, value: np.ndarray):
        self._base_indexes = value

    @property
    def total_cost_vector(self):
        return self._c

    @total_cost_vector.setter
    def total_cost_vector(self, value):
        self._c = value


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
    """Classe che rappresenta il problema lineare da ottimizzare

    Per quanto riguarda i vettori dei costi, ce ne sono due: normali e ridotti. Questi vengono immagazzinati allo stesso
    modo, ma l'handling della struttura dati e l'aggiornamento dei costi ridotti è implementato qui.
    Attenzione all'handling in quanto gli indici di base non sono qui ma nella constraint matrix.
    """

    def __init__(self, constr_matrix: ConstraintMatrix, b: np.ndarray, c: CostVector) -> None:
        self._previous_pivot = 0
        self.constraint_matrix = constr_matrix
        self._known_terms = b
        self.cost_vector = c
        self._reduced_cost_vector = CostVector(-c.total_cost_vector, c.base_indexes)

    @property
    def reduced_cost_base(self):
        return self._reduced_cost_vector.base_cost

    @reduced_cost_base.setter
    def reduced_cost_base(self, value: np.ndarray):
        self._reduced_cost_vector.non_base_cost = value

    @property
    def reduced_cost_non_base(self):
        return self._reduced_cost_vector.non_base_cost

    @reduced_cost_non_base.setter
    def reduced_cost_non_base(self, value: np.ndarray):
        self._reduced_cost_vector.non_base_cost = value

    @property
    def reduced_cost_total_vector(self):
        return self._reduced_cost_vector.total_cost_vector

    @reduced_cost_total_vector.setter
    def reduced_cost_total_vector(self, value: np.ndarray):
        self._reduced_cost_vector.total_cost_vector = value

    @property
    def solution_vertex(self):
        return np.linalg.solve(self.constraint_matrix.base_matrix, self._known_terms)

    @property
    def objective_value(self):
        return self.reduced_cost_base * np.linalg.inv(self.constraint_matrix.base_matrix) * self._known_terms

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
        entering_column_index = np.asarray(
            np.where(self.reduced_cost_non_base.T == np.amin(self.reduced_cost_non_base.T)))
        if entering_column_index.shape[0] == 2:
            return int(np.unique(np.asarray(entering_column_index[1])))
        elif entering_column_index.shape[0] == 1:
            return int(np.unique(np.asarray(entering_column_index[0])))
        else:
            raise IndexError("Number of dimensions in vector not recognized")

    @property
    def entering_column(self) -> np.ndarray:
        return np.reshape(np.linalg.solve(self.constraint_matrix.base_matrix,
                                          self.constraint_matrix.constraint_matrix[:, self.entering_column_index]),
                          self._known_terms.shape)

    @property
    def entering_variable_reduced_cost(self):
        return self.reduced_cost_non_base[self.entering_column_index]

    @property
    def current_pivot_element(self):
        return self.constraint_matrix.constraint_matrix[self.exit_variable_index, self.entering_column_index]

    @property
    def previous_pivot_element(self):
        return self._previous_pivot

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
                ratio.append({'ratio_value': float(self._known_terms[index] / d[index]), 'ratio_index': index})
        ratios = [x['ratio_value'] for x in ratio]
        min_ratio = min(ratios)
        exit_variable_index = [x['ratio_index'] for x in ratio if x['ratio_value'] == min_ratio]
        return int(exit_variable_index[0])

    @property
    def exit_variable_row(self):
        return self.constraint_matrix.base_matrix[self.exit_variable_index]

    def update_revised_simplex_constraint_matrix(self):
        """Questo metodo aggiorna tutto il tableau, in pratica ogni riga va aggiornata secondo le istruzioni del
        libro """
        self.constraint_matrix.constraint_matrix = self.constraint_matrix.constraint_matrix.astype('float')
        self._previous_pivot = self.current_pivot_element
        self.constraint_matrix.constraint_matrix[self.exit_variable_index, :] /= self.current_pivot_element
        multiplier = [idx for idx in range(0, len(self.entering_column)) if idx != self.exit_variable_index]
        for idx in multiplier:
            self.constraint_matrix.constraint_matrix[idx, :] = self.constraint_matrix.constraint_matrix[idx, :] - \
                                                               self.constraint_matrix.constraint_matrix[
                                                                   idx, self.entering_column_index] * \
                                                               self.constraint_matrix.constraint_matrix[
                                                               self.exit_variable_index,
                                                               :]

    def update_revised_simplex_known_terms(self):
        """
        Questo è da farsi allo stesso modo del pivoting. Questo lo devo aggiornare con la procedura di pivoting.
        :math:`b_i = b_i+\\frac{b_r}{a_{rk}}*c_{k}` cioè gli devo aggiungere :math:`c_k` volte la nuova r-esima riga,
        cioè quella uscente.

        """
        self._known_terms = self._known_terms.astype('float')
        self._known_terms = np.dot(self.constraint_matrix.base_matrix, self._known_terms)

    def update_revised_simplex_cost_vector(self):
        """
        Questo mi tocca gestirlo in 2 tempi:
            * Prima scambio i costi ridotti della variabile di base associata alla leaving row index :math:`X_{Br}`
            * Poi aggiungo reduced :math:`c_k` volte la r-esima riga aggiornata
        :return:
        """
        # TODO: TROVARE UN MODO MIGLIORE DI FARE QUESTO
        self.cost_vector._c = self.cost_vector._c.astype('float')
        self._reduced_cost_vector._c = self._reduced_cost_vector._c.astype('float')
        self.reduced_cost_total_vector = self.reduced_cost_total_vector.reshape(
            self.constraint_matrix.constraint_matrix.shape[1]) - \
                                         self.reduced_cost_non_base[
                                             self.entering_column_index] * self.constraint_matrix.constraint_matrix[
                                                                           self.exit_variable_index, :]

    def __str__(self):
        from prettytable import PrettyTable
        x = PrettyTable()
        x.field_names = ["Base variables", "Reduced NON base variables", "RHS solution"]
        x.add_row([self.reduced_cost_base.T, self.reduced_cost_non_base.T, np.unique(self.objective_value.T)])
        x.add_row(["---------------------", "---------------------", "---------------------"])
        x.add_row([self.constraint_matrix.base_matrix, self.constraint_matrix.non_base_matrix, self._known_terms])
        return x.__str__()

    def __next__(self):
        self.update_revised_simplex_constraint_matrix()
        self.update_revised_simplex_known_terms()
        self.update_revised_simplex_cost_vector()
        print(self)


if __name__ == '__main__':
    A = np.array([[1, 2, 3, 1, -3, 1, 0, 0], [2, -1, 2, 2, 1, 0, 1, 0], [-3, 2, 1, -1, 2, 0, 0, 1]])
    B = np.array([5, 6, 7])
    b = np.array([[9], [10], [11]])
    c = CostVector(np.array([[4], [3], [1], [7], [6], [0], [0], [0]]), B)
    constraint_matrix = ConstraintMatrix(A, B)
    N = np.array([[1, 2, 3, 1, -3], [2, -1, 2, 2, 1], [-3, 2, 1, -1, 2]])
    LP = LinearProgram(constraint_matrix, b, c)
    next(LP)

import numpy as np

# I PROBLEMI DEVONO ESSERE IN FORMA UGUAGLIANZA
from docplex.mp.model import Model


class LinearProgram:
    def __init__(self, A=np.empty([0, 0]), b=np.empty([0, 0]), c=np.empty([0, 0]), minmax="MAX"):
        self.A = A
        self.b = b
        self.c = c
        self.x = np.empty([len(c), 1])
        self.minmax = minmax
        self.optimalValue = None
        self.feasible = True
        self.bounded = False


class CostraintMatrix:
    """Represents the constraint matrix :math:`A`

        Questa classe rappresenta la matrice dei vincoli del problema. È stata aggiornata con metodi accessori per
        garantire la consistenza della matrice durante le varie operazioni. Secondo il motto
        "`We are all consenting adults <https://python-guide-chinese.readthedocs.io/zh_CN/latest/writing/style.html
        #we-are-all-consenting-adults>`_" possono essere modificati ma queste guardie proteggono contro accidentalità.

        .. Caution:: I PROBLEMI DEVONO ESSERE IN FORMA STANDARD. [notazione]_

        .. [notazione] Algorithms for Optimization, Mykel J. Kochenderfer, Tim A. Wheeler
        """

    def __init__(self, constraint_matrix: np.ndarray = np.empty([0, 0]), base_indexes: set = None) -> None:
        """
        Questo metodo inizializza le seguenti strutture dati:
            * Matrice dei vincoli totale del problema
            * Insieme degli indici che rappresentano le colonne della matrice di base
            * Insieme degli indici che rappresentano le colonne della matrice NON di base
        La cosa particolare è che l'insieme degli indici è fatto tramite un *mutable* dunque bisogna fare attenzione a
        come viene usato, in quanto se inizializzassi nuovamente l'insieme degli indici questo conserva lo stato come
        `qui <https://docs.python-guide.org/writing/gotchas/>`_. Per evitare, metto una guardia sul empty set.

        :param constraint_matrix: Matrice dei vincoli da dare in pasto alla classe CostraintMatrix
        :param base_indexes: Questa è la partizione iniziale che devi fornire se non vuoi risolvere un prima fase.
        """
        if base_indexes is None:
            base_indexes = set()
        self._A = constraint_matrix
        self._B = base_indexes
        if self._A.size != 0:
            self._indexes = set(range(1, A.shape[1]))
        else:
            self._indexes = set()
        self._N = self._indexes.difference(self._B)

    @property
    def non_base_indexes(self):
        self._N = self._indexes.difference(self._B)
        return self._N

    @non_base_indexes.setter
    def non_base_indexes(self, _):
        raise IndexError("Can't change non base indexes! You should touch only base indexes...")

    @property
    def base_matrix(self) -> np.ndarray:
        return self._A[self._B]

    @base_matrix.setter
    def base_matrix(self, _):
        raise IndexError("Can't change base matrix! You should touch only the total constraint matrix or indexes...")

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
        return self._A[self._N]

    @non_base_matrix.setter
    def non_base_matrix(self, value):
        raise IndexError(
            "Can't change non base matrix! You should touch only the total constraint matrix or indexes...")

    @property
    def base_indexes(self) -> set:
        return self._B

    @base_indexes.setter
    def base_indexes(self, new_base_indexes: set = None):
        if new_base_indexes is None:
            new_base_indexes = set()
        else:
            self._B = new_base_indexes


def get_vertex(B: np.array, LP: LinearProgram) -> np.matrix:
    """
    Restituisci il vertice

    Questa funzione gli viene dato ingresso il problema di programmazione lineare e una lista di indici che saranno le
    colonne della matrice dei vincoli che vanno a creare la nuova base. Dalla teoria sappiamo che questa matrice è un
    vertice del politopo che è la mia regione ammissibile. Questo è il X_b che è il sottogruppo delle variabili di base
    della soluzione.

    :param B: Lista delle colonne da estrarre e valutare
    :param LP: Problema comprensivo di matrice dei vincoli da cui estrarre il vertice
    :return: Vertice
    :raise  np.linalg.LinAlgError
    """
    A = LP.A
    b = LP.b
    B = np.sort(B)
    AB = A[:, B]
    vertex = np.linalg.solve(AB, b)
    return vertex


def row_pivot(LP: LinearProgram, B: np.array, q: int) -> tuple:
    """
    Questa funzione implementa la transizione da un vertice al successivo per la fase di ottimizzazione. Nel metodo del
    simplesso mi devo muovere da un vertice al successivo, qui uso l'euristica del MIN_RATIO. Questa operazione nei
    libri classici è detta [APPL1]_ PIVOTING. Per ogni variabile NON di base j e per ogni colonna a_j della matrice dei
    coefficienti NON di base associata a quel j, questa funzione calcola la colonna nello spazio ridotto (quello m+n-
    dimensionale). Inoltre si calcola il [WHLR]_ :math:`x_q` e cioè quel valore che porta la variabile più vicina al suo vincolo
    a sbatterci contro, cioè una delle variabili andrà a zero, è proprio quella indice q con valore associato x_q', in
    uscita avrò proprio questo.

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

    :param LP: Problema da ottimizzare
    :param B: Partizione degli indici per le variabili di base all'iterazione corrente
    :return: Coppia (indice della variabile in uscita, valore che la azzera)
    :param q: Indice della variable che entrerà in base
    """
    A = LP.A
    b = LP.b
    n = A.shape[1]
    B.sort()
    b_inds = B
    AB = A[:, b_inds]
    xB = np.linalg.solve(AB, b)
    Aq = A[:, q]
    d = np.linalg.solve(AB, Aq)
    p, xq = 0, np.inf
    for i in range(0, d.shape[0]):
        if d[i] > 0:
            v = xB[i] / d[i]
            if v < xq:
                p, xq = i, v
    return p, xq


def get_lambda(Ab: np.array, c: np.array):
    """
    Funzione usata per il calcolo dei moltiplicatori di Lagrange lambda. Nel caso di problemi lineari so che i
    moltiplicatori di Lagrange corrispondono alle variabili duali. Queste variabili duali le uso per calcolarmi il
    certificato duale a fine ottimizzazione. In pratica ciò che faccio è moltiplicare il vettore dei costi delle
    variabili di base con la matrice dei vincoli delle variabili di base.

    Calcolo delle variabili duali: :math:`c^T B^{-1} = u^T`

    # TODO:FA CAGARE CHIAMARLA COSì VEDI IL MAIN...

    >>> import numpy as np
    >>> c = np.array([[3], [-1], [0], [0]])
    >>> A = np.array([[1, 1, 1, 0], [-4, 2, 0, 1]])
    >>> B = np.array([2, 3])
    >>> get_lambda(A[:, B], c[B])
    array([[0.],
           [0.]])

    :param Ab: Matrice di base da usare
    :param c: Vettore dei costi del problema
    :return: Vettore dei moltiplicatori di Lagrange lambda
    """

    return np.linalg.solve(np.matrix.transpose(Ab), c)


def get_mu_v(cv: np.array, Ab: np.array, Av: np.array, cb: np.array) -> np.array:
    """
    Funzione che restituisce il secondo vettore di moltiplicatori di Lagrange (quello associato ai vincoli di non
    negatività). Questo vettore nella dimostrazione di [WHLR]_ viene scomposto fra costi delle variabili di base (che
    viene posto a zero) e quelli NON di base (che sono oggetto diu questa funzione). In pratica nel libro [APPL1]_ si
    tratta dei costi ridotti (cioè :math:`\mu_v  \in R^{m+n}` ). Questi costi li ispezionerò per controllare
    l'ottimalità della soluzione corrente, se c'è anche sola una componente negativa, questa viola la duale
    ammissibilità e quindi necessariamente non può essere soluzione ottima.

    .. [WHLR] Wheeler, Algorithm for optimization sezione 11.2.2
    .. [APPL1] libro Applied Integer programming pag. 234-235 Algoritmo Revised Simplex
    :param cv: Sotto-vettore dei costi delle variabili non di base
    :param Ab: Matrice di base
    :param Av: Matrice NON di base (colonne associate alle variabili non di base, nel tableau)
    :param cb: Sotto-vettore delle variabili di base
    :return: Vettore associato alle variabili NON di base dei moltiplicatori associato ai vincoli di non negatività
    """
    reduced_cost = np.matmul(np.invert(Ab), Av)
    base_cost = np.dot(np.transpose(reduced_cost), cb)
    mu_v = cv - base_cost
    return mu_v


def step_LP(B: np.array, LP: LinearProgram):
    """
    # TODO: sudiividere il metodo, troppo complesso così. Metti almeno un __init__ forse è il caso mettere una CLASS
    Questa funzione restituisce la nuova matrice di base nella iterazione della fase di ottimizzazione. NON restituisco:
        * Il nuovo vertice
        * Vettore dei moltiplicatori associati alla matrice A dei vincoli (lambda)
        * Vettore dei moltiplicatori associato ai vincoli di non negatività (non di base)

    :param B:   Matrice di base alla iterazione precedente
    :param LP:  Problema da ottimizzare
    :return:
    """
    A = LP.A
    b = LP.b
    c = LP.c
    n = A.shape[1]
    B.sort()
    b_inds = B
    B_set = set(B)
    x_set = set(range(1, n + 1))
    n_inds = sorted(x_set.difference(B_set))
    AB = A[:, b_inds]
    Av = np.delete(A, b_inds, axis=1)
    xB = np.linalg.solve(AB, b)
    cB = c[b_inds]
    lambda_mul = np.linalg.solve(np.transpose(AB), cB)
    cV = np.delete(c, b_inds)
    reduced_cost = np.dot(np.invert(AB), Av)
    mu_v = cV - np.dot(np.transpose(reduced_cost), lambda_mul)
    mu_v = np.unique(-mu_v)


def column_pivot(mu_v=None, Ab=np.ndarray):
    q, p, xq, delta = 0, 0, np.inf, np.inf
    candidate_pivots = np.array([elem for elem in mu_v if elem < 0])
    column_index = np.where(candidate_pivots == np.amin(candidate_pivots))
    reduced_space_column = np.linalg.solve(Ab, A[column_index])
    reduced_cost = np.dot(c_b)
    return column_index


def minimize_lp(B, LP):
    done = False
    while not done:
        B, done = step_LP(B, LP)
    return B


def solve_LP_no_first_base(B: np.array, LP: LinearProgram):
    A = LP.A
    b = LP.b
    c = LP.c
    m, n = A.shape
    z = np.ones(m)
    lista = [+1 if b[j] >= 0 else -1 for j in b if b[j] >= 0]
    a = np.array(lista)
    Z = np.diag(a)
    A_concat = np.concatenate(A, Z, 1)
    b_new = b
    c_new = np.concatenate(np.zeros(n), z, 0)
    LP_init = LinearProgram(A_concat, b_new, c_new)
    B = np.arange(start=1, stop=m + 1, step=1) + n

    for i in B:
        if B[i] > n:
            raise ValueError("Infeasible")

    solve_LP_no_first_base(B, LP_init)


if __name__ == '__main__':
    """ Qui testo l'algoritmo dall'inzio alla fine..."""
    A = np.array([[1, 2, 3, 1, -3, 1, 0, 0], [2, -1, 2, 2, 1, 0, 1, 0], [-3, 2, 1, -1, 2, 0, 0, 1]])
    b = np.array([[9], [10], [11]])
    x = np.empty([8, 1])
    c = np.array([[4], [3], [1], [7], [6], [0], [0], [0]])
    LP = LinearProgram(A, b, c)
    B = np.array([5, 6, 7])
    print("First vertex: ")
    print(get_vertex(B, LP))
    print("Lambda: ")
    print(get_lambda(LP.A[:, B], c[B]))
    print("Reduced costs: ")
    mu_s = get_mu_v(np.delete(c, B), LP.A[:, B], np.delete(A, B, axis=1), c[B])
    print(np.unique(mu_s, axis=0))
    B, done = step_LP(B, LP)

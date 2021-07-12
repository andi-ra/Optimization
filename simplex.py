import numpy as np


# I PROBLEMI DEVONO ESSERE IN FORMA UGUAGLIANZA

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


def get_vertex(B: np.array, LP: LinearProgram) -> np.matrix:
    """
    Restituisci il vertice

    Questa funzione gli viene dato ingresso il problema di programmazione lineare e una lista di indici che saranno le
    colonne della matrice dei vincoli che vanno a creare la nuova base. Dalla teoria sappiamo che questa matrice è un
    vertice del politopo che è la mia regione ammissibile.

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


def edge_transition(LP: LinearProgram, B: np.array, q: int):
    """
    Questa funzione implementa la transizione da un vertice al successivo per la di ottimizzazione. Nel metodo del
    simplesso mi devo muovere da un vertice al successivo, qui uso l'euristica del MIN_RATIO.

    :param q: Indice della variable che uscirà di base
    """
    A = LP.A
    b = LP.b
    n = A.shape[1]
    B.sort()
    b_inds = B
    B_set = set(B)
    x_set = set(range(1, n + 1))
    if len(x_set) >= len(B_set):
        n_inds = sorted(x_set.difference(B_set))
    else:
        raise ValueError("Dimension mismatch in set difference")
    AB = A[:, b_inds]
    # A[:, n_inds[q - 1]] questa è Av
    xB = np.linalg.solve(AB, b)
    Av = np.delete(A, b_inds, axis=1)
    d = np.linalg.solve(AB, Av)
    p, xq = 0, np.inf
    for i in range(0, d.shape[0]):
        for j in range(0, d.shape[1]):
            if d[i][j] > 0:
                v = xB[i] / d[i][j]
                if v < xq:
                    p, xq = i, v
    return p, xq


def get_lambda(Ab: np.array, c: np.array):
    """
    Funzione usata per il calcolo dei moltiplicatori di Lagrange lambda. Nel caso di problemi lineari so che i
    moltiplicatori di Lagrange corrispondono alle variabili duali.

    :param Ab: Matrice di base da usare
    :param c: Vettore dei costi del problema
    :return: Vettore dei moltiplicatori di Lagrange lambda
    """
    return np.linalg.solve(np.matrix.transpose(Ab), c)


def get_mu_v(cv: np.array, Ab: np.array, Av: np.array, cb: np.array) -> np.array:
    """
    Funzione che restituisce il secondo vettore di moltiplicatori di Lagrange (quello associato ai vincoli di non
    negatività). Questo vettore nella dimostrazione di Weeler viene scomposto fra costi delle varibili di base (che
    viene posto a zero) e quelli NON di base (che sono oggetto diu questa funzione).

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
    lambda_mul = np.linalg.solve(AB, cB)
    cV = np.delete(c, b_inds)
    mu_v = cV - np.dot(np.transpose(Av), lambda_mul)
    q, p, xq, delta = 0, 0, np.inf, np.inf
    for i in range(0, mu_v.shape[0]):
        for j in range(0, mu_v.shape[1]):
            if mu_v[i][j] < 0:
                pi, xi = edge_transition(LP, B, i)
                if mu_v[i][j] * xi < delta:
                    q, p, xq, delta = i, pi, xi, mu_v[i][j] * xi
            if q == 0:
                return (B, True)
    if np.isinf(xq):
        raise ValueError("Unbounded problem")
    j = next(index for index in B if b_inds[p] == B[index])
    B[j] = n_inds[q]  # swap indices
    return (B, False)


def minimize_lp(B, LP):
    done = False
    while not done:
        B, done = step_LP(B, LP)
    return B


if __name__ == '__main__':
    # Diet problem example
    # Ricordati che nei libri gli indici partono da 1 QUI PARTONO DA 0 (sottrai 1)
    A = np.array([[1, 1, 1, 0], [-4, 2, 0, 1]])
    b = np.array([[9], [2]])
    x = np.empty([4, 1])
    c = np.array([[6], [3], [0], [2]])
    LP = LinearProgram(A, b, c)
    B = np.array([2, 3])
    minimize_lp(B, LP)
    AB = A[:, B]
    xB = np.linalg.solve(A, b)
    print(xB)

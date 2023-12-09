from src.criterions import draw_criterion
from src.em import em_algorithm
from src.graphs import draw_graph, random_graph
from src.utils import sort_parameters

if __name__ == "__main__":
    # Test the algorithm on a random graph
    Q = 10
    n = 100
    X, Z, alpha, pi = random_graph(n, Q)
    alpha, pi = sort_parameters(alpha, pi)

    draw_graph(X, Z)
    draw_criterion(n)

    estimated_alpha, estimated_pi, tau = em_algorithm(X, Q, n_init=10, iterations=100)

    print(estimated_alpha)
    print(alpha)

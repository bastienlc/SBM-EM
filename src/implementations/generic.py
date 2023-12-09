class GenericImplementation:
    def __init__(self):
        pass

    def e_step(self, X, alpha, pi):
        pass

    def init_tau(self, n, Q):
        pass

    def init_X(self, X):
        pass

    def m_step(self, X, tau):
        pass

    def log_likelihood(self, X, alpha, pi, tau):
        pass

    def parameters_are_ok(self, alpha, pi, tau):
        pass

    def fixed_point_iteration(self, tau, X, alpha, pi):
        pass

    def input(self, array):
        pass

    def output(self, array):
        pass

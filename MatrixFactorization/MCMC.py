


class GibbsSamplerPMF(object):
    def __init__(self, R, n):
        self.R = R
        self.n = n

    # Perform T steps, in each step, sample the parameters from the distribution
    # as specified
    def learn_matrix_factorization(self, T):
        self.initialize_UV()
        self.T = T
        for t in range(T):
            pass

    def sample_hyperparameter(self):
        
        pass

    def sample_user_feature(self):
        pass

    def sample_movie_feature(self):
        pass

    def initialize_UV(self):

        pass


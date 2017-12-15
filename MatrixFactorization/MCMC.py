import numpy as np
import collections
from numpy import dot
from numpy.linalg import inv
from scipy.stats import wishart
from scipy.stats import norm
from scipy.stats import gamma
from numpy.random import multivariate_normal


class GibbsSamplerPMF(object):
    def __init__(self, R, n):
        self.R = R
        self.D = n
        self.beta0 = 0.1 # is this a good choice?
        self.W0 = np.eye(n)
        self.v0 = n
        self.mu0 = np.zeros((n,))
        self.tau = 1 # or 1
        self.a_tau = 2
        self.b_tau = 0

        self.N = len(R)
        self.M = len(R[0])
        self.rated_movie_by_user = collections.defaultdict(list)
        self.rated_user_by_movie = collections.defaultdict(list)

        self.U = np.random.randn(self.N, self.D)
        self.V = np.random.randn(self.M, self.D)
        self.K = 0

        self.find_ratings_by_movies_user()
        self.R_predict = np.zeros_like(R)
        self.num_sample = 0

    def find_ratings_by_movies_user(self):
        for user, row in enumerate(self.R):
            for movie, score in enumerate(row):
                if score:
                    self.K += 1 # K is the number of observed entries
                    self.rated_movie_by_user[user].append(movie)
                    self.rated_user_by_movie[movie].append(user)

    # Perform T steps, in each step, sample the parameters from the distribution
    # as specified
    # Each call to this function generates a new sample
    def learn_matrix_factorization(self):
        self.num_sample += 1
        Mu_U, Lambda_U = self.sample_user_hyperparameter()
        Mu_V, Lambda_V = self.sample_movie_hyperparameter()

        # Sample user vectors
        for i in range(self.N):
            self.U[i, :] = self.sample_user_feature(Mu_U, Lambda_U, i)

        # Sample movie vectors
        for j in range(self.M):
            self.V[j, :] = self.sample_movie_feature(Mu_V, Lambda_V, j)
        # Predict
        R_predict = dot(self.U, self.V.T)

        # Sample alpha (precision)
        self.tau = self.sample_alpha(R_predict)

        # Accummulate prediction
        self.R_predict += R_predict

    def sample_alpha(self, R_predict):
        # return 2
        au = self.a_tau + self.K
        bu = self.b_tau
        for i in range(self.N):
            for j in self.rated_movie_by_user[i]:
                bu += self.tau/2 * (self.R[i][j]-R_predict[i][j])**2
        sample = np.random.gamma(au, 1/bu)
        return sample

    def sample_user_hyperparameter(self):
        U_bar = np.mean(self.U, axis=0)
        S_bar = np.zeros((self.D, self.D))
        for i in range(self.N):
            S_bar += np.outer(self.U[i, :], self.U[i, :])
        beta0 = self.beta0 + self.N
        mu0 = (self.beta0 * self.mu0 + self.N * U_bar)/beta0
        v0 = self.v0 + self.N

        # Note that our choice of W0 is the identity matrix so inverse has no difference
        W0_inverse = inv(self.W0) + S_bar + (self.beta0*self.N)/(self.beta0+self.N)\
                                            *np.outer(self.mu0-U_bar, self.mu0-U_bar)
        W0 = inv(W0_inverse)
        # Sample Lambda_U from Wishart
        Lambda_U = wishart.rvs(v0, W0)

        # Sample muU from Gaussian
        precision = inv(beta0 * Lambda_U)
        Mu_U = multivariate_normal(mean=mu0, cov=precision)

        return Mu_U, Lambda_U

    def sample_movie_hyperparameter(self):
        V_bar = np.mean(self.V, axis=0)
        S_bar = np.zeros((self.D, self.D))
        for i in range(self.M):
            S_bar += np.outer(self.V[i, :], self.V[i, :])

        beta0 = self.beta0 + self.M
        mu0 = (self.beta0 * self.mu0 + self.M * V_bar)/beta0
        v0 = self.v0 + self.M
        W0_inverse = inv(self.W0) + S_bar + (self.beta0 * self.M)/(self.beta0+self.M)\
                                            * np.outer(self.mu0-V_bar, self.mu0-V_bar)
        W0 = inv(W0_inverse)

        # Sample Lambda_U from Wishart
        Lambda_V = wishart.rvs(v0, W0)
        # Sample muU from Gaussian
        precision = inv(beta0 * Lambda_V)
        Mu_V = multivariate_normal(mean=mu0, cov=precision)

        return Mu_V, Lambda_V

    def sample_user_feature(self, Mu_u, Lambda_u, i):
        Lambda_i = np.copy(Lambda_u)
        mu_i = dot(Lambda_u, Mu_u)
        for j in self.rated_movie_by_user[i]:
            Lambda_i += self.tau * np.outer(self.V[j, :], self.V[j, :])
            mu_i += self.tau * self.V[j, :] * self.R[i][j]
        Lambda_inv = inv(Lambda_i)
        mu_i = dot(Lambda_inv, mu_i)
        sample = multivariate_normal(mu_i, Lambda_inv)
        return sample

    def sample_movie_feature(self, Mu_v, Lambda_v, j):
        Lambda_j = np.copy(Lambda_v)
        mu_j = dot(Lambda_v, Mu_v)

        for i in self.rated_user_by_movie[j]:
            Lambda_j += self.tau * np.outer(self.U[i, :], self.U[i, :])
            mu_j += self.tau * self.U[i, :] * self.R[i][j]

        Lambda_inv = inv(Lambda_j)
        mu_j = dot(Lambda_inv, mu_j)
        sample = multivariate_normal(mu_j, Lambda_inv)
        return sample

    def evaluate(self, users, movies, scores):
        R_predict = self.R_predict/self.num_sample # Take the average
        loss = 0.0
        for i in range(len(users)):
            user = users[i]
            movie = movies[i]
            score = scores[i]

            loss += (R_predict[user][movie]-score)**2
        return loss/len(users)

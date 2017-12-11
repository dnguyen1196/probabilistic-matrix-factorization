"""
Expectation-Maximization algorithm to perform matrix factorization
E step: X = W.M+(1-W).M~
M step: [U,V] = svd(X)n
        M~ = UV'
"""
import numpy as np
from numpy import multiply
from numpy import dot


class EM_SVD(object):
    def __init__(self, M, n):
        self.M = M
        self.n = n # Rank of matrix factorization
        self.I = len(M) # Number of users
        self.J = len(M[0]) # Number of movies
        self.W = np.zeros((self.I, self.J))
        for i, row in enumerate(self.M):
            for j, score in enumerate(row):
                if score:
                    self.W[i][j] = 1

        self.max_iters = 100
        self.U, _, self.V = np.linalg.svd(self.M, full_matrices=False)
        self.M_ = np.dot(self.U, self.V)

    def learn_matrix_factorization(self):
        X = multiply(self.W, self.M)+multiply((1-self.W),self.M_)
        self.U, _, self.V = np.linalg.svd(X,full_matrices=False)
        M_ = dot(self.U, self.V)
        norm_dif = self.check_stopping_condition(M_)
        self.M_ = M_
        return norm_dif

    def check_stopping_condition(self, M_):
        return (np.abs(M_-self.M_)).max()

    def evaluate(self, users, movies, scores):
        err = 0.0
        count = 0
        assert(len(users)==len(movies)==len(scores))
        for i in range(len(users)):
            user = users[i]
            movie = movies[i]
            score = scores[i]
            if score:
                predict = self.M_[user][movie]
                err += (predict-score)**2
                count += 1
        return err/count


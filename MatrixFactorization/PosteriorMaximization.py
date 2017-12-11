"""
Perform gradient descent to optimize he posterior
"""
import numpy as np
import collections
from numpy.linalg import norm


class MAP(object):
    def __init__(self, R, n, K=5, sv=1, su=1, s=1, lr=0.01):
        self.n = n
        self.I = len(R)
        self.J = len(R[0])
        self.K = K
        self.sv = sv # Variance for V
        self.su = su # Variance for U
        self.s = s # Variance for u'v
        self.lv = s/sv
        self.lu = s/su
        self.R = np.zeros_like(R)
        self.rated_movie_by_user = collections.defaultdict(list)
        self.lr = lr # Learning rate

        self.U = np.random.randn(self.I,self.n)
        self.V = np.random.randn(self.J,self.n)

        self.U_change = np.zeros((self.I,))
        self.V_change = np.zeros((self.J,))

        # Transform the rating so that all ratings are in [0,1]
        for i, row in enumerate(R):
            for j, score in enumerate(row):
                if score:
                    self.rated_movie_by_user[i].append(j)
                    self.R[i][j] = self.t(score)

    def t(self,x):
        return (float(x)-1)/(self.K-1)

    # Perform one step of matrix factorization
    def learn_matrix_factorization(self):
        v_change = np.zeros_like(self.V)
        for i in range(self.I):
            ui = self.U[i, :]
            delta_ui = np.zeros((self.n,))
            for j in self.rated_movie_by_user[i]:
                vj = self.V[j, :]
                uv = np.dot(ui,vj)
                guv = 1/(1+np.exp(-uv))
                # ui = ui + lr * (g(uv) - r)g(uv)(1-g(uv))v
                # vj = vj + lr * (g(uv) - r)g(uv)(1-g(uv))u
                # Note that for each u'v, there is contribution from u and v
                delta = (self.R[i][j]-guv)*(-guv)*(1-guv)
                delta_ui += self.lr * delta * vj
                v_change[j, :] += self.lr* delta * ui
                # Use V_update to store updates to V, since note that we're not
                # dealing with the same user i again

            ui_update = delta_ui + self.lr*self.lu*ui
            self.U_change[i] = norm(ui_update)
            self.U[i, :] -= ui_update

        v_change = v_change + self.lr * self.lv * self.V
        self.V_change = norm(v_change, axis=0)
        self.V -= v_change

    def evaluate_objective_function(self):
        error = 0.0
        for i, row in enumerate(self.R):
            for j, score in enumerate(row):
                if score:
                    ui = self.U[i, :]
                    vj = self.V[j, :]
                    uv = np.inner(ui,vj)
                    s = 1/(1 + np.exp(-uv))
                    error += (score - s)**2
        for i in range(self.I):
            error += self.lu * np.linalg.norm(self.U[i, :])**2
        for j in range(self.J):
            error += self.lv * np.linalg.norm(self.V[j, :])**2
        return error

    def check_stopping_condition(self):
        return max(max(self.U_change), max(self.V_change))

    def evaluate(self, users, movies, scores):
        err = 0.0
        count = 0
        assert(len(users)==len(movies)==len(scores))
        for i in range(len(users)):
            user = users[i]
            movie = movies[i]
            score = scores[i]
            if score:
                predict = self.predict_rating(user, movie)
                err += (predict-score)**2
                count += 1
        return err/count

    def predict_rating(self, user, movie):
        return np.dot(self.U[user,:], self.V[movie,:])*(self.K-1)+1
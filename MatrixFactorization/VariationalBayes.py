"""
Variational Bayes method in Matrix completion
"""
import numpy
import collections
from numpy.linalg.linalg import inv
from numpy import outer
from numpy import diag
from numpy import dot
from numpy.linalg import norm
import logging
logger = logging.getLogger("dev")


class VBMatrixFactorization(object):
    def __init__(self, R, n, init="Random"):
        self.R = R # The rating matrix
        self.n = n # The rank of the factorization
        self.I = len(R) # Number of users
        self.J = len(R[0]) # Number of movies
        self.rho = numpy.ones([n,], dtype=float) / self.n
        self.sigma = numpy.ones([n,],dtype=float)
        self.tau = 1
        self.rated_movie_by_user = collections.defaultdict(list)
        self.find_ratings_by_movies_user()

        self.Psi = [diag(self.rho) for _ in range(self.J)]
        self.Phi = [diag(self.sigma) for _ in range(self.I)]

        self.U = numpy.random.randn(self.I,self.n)
        self.V = numpy.random.randn(self.J,self.n)
        self.S = [diag(1/self.rho) for _ in range(self.J)]

        self.u_convergece = numpy.ones([self.I,])
        self.v_convergence = numpy.ones([self.J,])

    # Find the list of rated movies by user as well as rated user by movie
    def find_ratings_by_movies_user(self):
        self.K = 0
        for user, row in enumerate(self.R):
            for movie, score in enumerate(row):
                if score:
                    self.K += 1 # K is the number of observed entries
                    self.rated_movie_by_user[user].append(movie)

    # Perform coordinate descent to learn matrix factorization
    def learn_matrix_factorization(self):
        # Initialize Sj and tj but note that Sj is initialized implicitly
        t = [numpy.zeros([self.n,]) for _ in range(self.J)] # initialize tj = 0, Sj
        for j in range(self.J): # Initialize Sj = diag(1/pl)
            self.S[j] = diag(1/self.rho)

        # for i = 1:I, update Phi_i and ui
        for i in range(self.I):
            self.compute_phi(i) # Compute Phi_i
            self.update_u(i) # Update ui
            # For j in N(i)
            for j in self.rated_movie_by_user[i]:
                # Sj = Sj + (phi+uu')/tau
                self.S[j] = self.S[j]+(self.Phi[i]+outer(self.U[i,:],self.U[i,:]))/self.tau
                # tj = tj+m_ij ui/tau
                t[j] += self.R[i][j]*self.U[i,:]/self.tau

        # Update sigma
        self.update_sigma()

        # For j = 1:J update Q(vj)
        for j in range(self.J):
            self.Psi[j] = inv(self.S[j])
            self.update_v(j, t)

        # Update tau
        self.update_tau()

    def update_sigma(self):
        diag_sum = numpy.zeros((self.n,))
        for i in range(self.I):
            diag_sum += diag(self.Phi[i])
        column_sum = numpy.zeros((self.n,))
        for i in range(self.I):
            column_sum += numpy.power(self.U[i,:],2)
        self.sigma = 1/(self.I-1)*(diag_sum+column_sum)

    # TODO: figure out what is wrong with update_tau here
    def update_tau(self):
        total = 0.0
        for i in range(self.I):
            for j in self.rated_movie_by_user[i]:
                mij = self.R[i][j]
                ui = self.U[i, :]
                vj = self.V[j, :]
                uiui = numpy.outer(ui,ui)
                vjvj = numpy.outer(vj,vj)
                hadamard = numpy.multiply(self.Phi[i]+uiui, self.Psi[j]+vjvj)
                trace = hadamard.sum()
                total += mij**2-2*mij*dot(ui,vj) + trace
        self.tau = 1.0/(self.K-1)*total

    # Function to check for stopping condition
    def check_stopping_condition(self):
        return max(max(self.u_convergece), max(self.v_convergence))

    # Update ui
    def update_u(self, i):
        ui = numpy.zeros([self.n,])
        for j in self.rated_movie_by_user[i]:
            ui += self.R[i][j] * self.V[j,:] / self.tau
        update = dot(self.Phi[i], ui)
        self.u_convergece[i] = norm(update - self.U[i,:])
        self.U[i,:] = update

    # Compute phi_i
    def compute_phi(self, i):
        phi = diag(self.sigma)
        for j in self.rated_movie_by_user[i]:
            vj = self.V[j,:]
            phi = phi + (self.Psi[j] + outer(vj, vj)) / self.tau
        self.Phi[i] = inv(phi)

    # Update vj
    def update_v(self, j, t):
        update = dot(self.Psi[j], t[j])
        self.v_convergence[j] = norm(update-self.V[j,:])
        self.V[j,:] = update

    # Perform testing on some unobserved ratings
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
        return numpy.dot(self.U[user,:], self.V[movie,:])


import preprocess.Preprocess as DataCleaner
from MatrixFactorization.VariationalBayes import VBMatrixFactorization
from MatrixFactorization.ExpectationMaximization import EM_SVD
from MatrixFactorization.PosteriorMaximization import MAP
from MatrixFactorization.MCMC import GibbsSamplerPMF
import big_matrix.Matrix as Matrix
import os

import logging
logger = logging.Logger("dev")

raw_data_folder = os.path.join(os.getcwd(),"ml-100k")
clean_data_folder = os.path.join(os.getcwd(), "cleaned_data")

data_file = os.path.join(raw_data_folder, "ua.base")
data_cleaned = os.path.join(clean_data_folder, "ua.base.clean")

test_file = os.path.join(raw_data_folder, "ua.test")
test_cleaned = os.path.join(clean_data_folder, "ua.test.clean")

# nums_user = 943
# nums_movie = 1682
# DataCleaner.clean_user_rating_data(data_file, data_cleaned, nums_user, nums_movie)
# DataCleaner.clean_user_rating_data(test_file, test_cleaned, nums_user, nums_movie)

users, movies, scores = DataCleaner.build_test_data(test_cleaned)
R = Matrix.FormRatingMatrix(data_cleaned)

max_iters = 75
n = 15 # Modify this rank to obtain different factorization

EM = EM_SVD(R,n)
VB = VBMatrixFactorization(R, n)
MAP_MF = MAP(R, n)
Sampler = GibbsSamplerPMF(R, n)

print("Rank: ", n)
print("EM, VB, MAP, Gibbs")

for it in range(max_iters):
    EM.learn_matrix_factorization()
    EM_mse = EM.evaluate(users, movies, scores)
    print(EM_mse, end=" ")

    # VB.learn_matrix_factorization()
    # VB_mse = VB.evaluate(users, movies, scores)
    # print(VB_mse,end=" ")

    # MAP_MF.learn_matrix_factorization()
    # MAP_mse = MAP_MF.evaluate(users, movies, scores)
    # print(MAP_mse,end=" ")

    # Sampler.learn_matrix_factorization()
    # Gibbs_mse = Sampler.evaluate(users, movies, scores)
    # print(Gibbs_mse,end=" ")

    print()

import preprocess.Preprocess as DataCleaner
from MatrixFactorization.VariationalBayes import VBMatrixFactorization
from MatrixFactorization.ExpectationMaximization import EM_SVD
from MatrixFactorization.MaximumAPriori import MAP_original
from MatrixFactorization.PosteriorMaximization import MAP
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

max_iters = 200
n = 5

# EM = EM_SVD(R,n)
VB = VBMatrixFactorization(R, n)
# MAP_MF = MAP(R, n)
MAP_convex = MAP_original(R, n)

for it in range(max_iters):
    logging.warning(["iteration: ", it])
    # conv = EM.learn_matrix_factorization()
    # logging.warning(["Convergence: ", conv])
    # logging.warning(["MSE: ", EM.evaluate(users,movies,scores)])

    # VB.learn_matrix_factorization()
    # logging.warning(["Convergence: ", VB.check_stopping_condition()])
    # logging.warning(["MSE: ", VB.evaluate(users,movies,scores)])

    # MAP_MF.learn_matrix_factorization()
    # logging.warning(["Convergence: ", MAP_MF.check_stopping_condition()])
    # logging.warning(["Error: ", MAP_MF.evaluate_objective_function()])
    # logging.warning(["MSE: ", MAP_MF.evaluate(users, movies, scores)])

    MAP_convex.learn_matrix_factorization()
    logging.warning(["Convergence: ", MAP_convex.check_stopping_condition()])
    logging.warning(["Error: ", MAP_convex.evaluate_objective_function()])
    logging.warning(["MSE: ", MAP_convex.evaluate(users, movies, scores)])

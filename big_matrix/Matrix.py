import numpy


# Load the data intro a matrix from data file
def FormRatingMatrix(data_file):
    # Count the number of users and number of movies
    f = open(data_file, "r")
    dims = [int(dim) for dim in f.readline().strip().split(",")]
    nums_user = dims[0]
    nums_movie = dims[1]
    # Create sparse form of the matrix?
    rmatrix = numpy.empty([nums_user, nums_movie])
    for line in f:
        info = [int(d) for d in line.strip().split(",")]
        user_id = info[0]-1
        movie_id = info[1]-1
        rating = info[2]
        rmatrix[user_id][movie_id] = rating
    f.close()
    return rmatrix

"""
preprocessing module
"""


def clean_user_rating_data(filename, output_file, nums_user, nums_movie):
    f = open(filename, "r")
    out = open(output_file, "w")
    out.write(str(nums_user)+","+str(nums_movie)+"\n")
    for line in f:
        info = [data for data in line.strip().split("\t") if not data.isspace()]
        out.write(','.join(info[:-1])+"\n")
    f.close()
    out.close()


def build_test_data(filename):
    f = open(filename, "r")
    users = []
    movies = []
    scores = []
    f.readline()
    for line in f:
        info = [int(data) for data in line.strip().split(",")]
        users.append(info[0]-1)
        movies.append(info[1]-1)
        scores.append(info[2])
    f.close()
    return (users,movies, scores)
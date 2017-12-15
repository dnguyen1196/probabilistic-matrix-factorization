import matplotlib.pyplot as plt
import os

result_folder = "./results"

rank_5_results = "results_5"
rank_10_results = "results_10"
rank_15_results = "results_15"

em_5_results = "em_5"
em_10_results = "em_10"
em_15_results = "em_15"

em_results = [em_5_results, em_10_results, em_15_results]
other_results = [rank_5_results, rank_10_results, rank_15_results]

rank = 2
em_file = os.path.join(result_folder, em_results[rank])
other_file = os.path.join(result_folder, other_results[rank])

f1 = open(em_file)
f2 = open(other_file)

em_plot = []
for line in f1:
    em_plot.append(float(line.strip()))
f1.close()

VB_plot = []
MAP_plot = []
Gibbs_plot = []
for line in f2:
    vb, MAP, gibbs = line.strip().split(" ")
    VB_plot.append(float(vb))
    MAP_plot.append(float(MAP))
    Gibbs_plot.append(float(gibbs))
f2.close()

plt.plot(em_plot, label="EM")
plt.plot(VB_plot, label="VB")
plt.plot(MAP_plot, label="MAP")
plt.plot(Gibbs_plot, label="Gibbs")
plt.legend(loc="upper right")
plt.ylabel("test MSE")
plt.xlabel("iterations")
plt.show()
import matplotlib.pyplot as plt
import numpy as np
def load_ratings(filename):
  ratings = []
  with open(filename, "r") as f:
    line = f.readline()
    while line is not None and line != "":
      arr = line.split("\t")
      ratings.append(int(arr[2]))
      line = f.readline()
  return ratings


ratings = load_ratings("Data/ml-1m.train.rating")
plt.hist(ratings, bins=[1, 2, 3, 4, 5, 6])
plt.show()
print("Mean rating:", np.mean(ratings))
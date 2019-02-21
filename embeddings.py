import pickle
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from util import perfTSNE, plotPoints, getMovieIds, getIdsEmbeddings, get_ncf_id_to_movie_map


def load_emb(algo='GMF', type='item'):
  with open('Embeddings/%s_%s_embs.pkl' % (algo, type), 'rb') as pkl_file:
    embs = pickle.load(pkl_file)
    return embs

def get_distance_matirx(embs_vectors):
  return distance_matrix(embs_vectors, embs_vectors) # Minkowski distance with p-norm=2

def get_n_closest_movies(distance_matrix, id_to_movie, n=10):
  closest_movies_map = dict()
  for id, movie in id_to_movie.items():
    distances = distance_matrix[id]
    closest_movies = sorted(range(len(distances)), key=lambda k: distances[k])
    closest_movies_map[id] = closest_movies[:n]
  return closest_movies_map

# item_embs = load_emb('GMF')
# distance_matrix_gmf = get_distance_matirx(item_embs)
id_to_movie_map = get_ncf_id_to_movie_map()
# closest_movies_map = get_n_closest_movies(distance_matrix_gmf, id_to_movie_map)
# print(distance_matrix.shape)
with open('10_closest_movies.pkl', 'rb') as pkl_file:
  closest_movies_map = pickle.load(pkl_file)

with open("closest_movie_embeddings.txt","w+") as f:
  for k, v in closest_movies_map.items():
    for mv in v:
      if mv in id_to_movie_map:

        f.write((" ".join(id_to_movie_map[mv])))
    f.write("----\n")


# X_embedded = perfTSNE(item_embs)





# plotPoints(X_embedded, 'g.', "all")
#
# horror_ids = getMovieIds(["Horror"])
# horror_embedded_vectors = getIdsEmbeddings(X_embedded, horror_ids)
# plotPoints(horror_embedded_vectors, 'r.', "horror")
#
# # scifi_item_embs = getMovieIds(["Fantasy"])
# # scifi_embedded_vectors = getIdsEmbeddings(X_embedded, scifi_item_embs)
# # plotPoints(scifi_embedded_vectors, 'b.', "scifi")
#
# hns_item_embs = getMovieIds(["Animation"])
# hns_embedded_vectors = getIdsEmbeddings(X_embedded, hns_item_embs)
# plotPoints(hns_embedded_vectors, 'y.', "scifi-horror")
# plt.show()

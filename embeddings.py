import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from util import perfTSNE, plotPoints, getIdsEmbeddings, \
  get_users, load_emb

def get_distance_matirx(embs_vectors):
  return distance_matrix(embs_vectors, embs_vectors) # Minkowski distance with p-norm=2

def get_n_closest_items(distance_matrix, id_to_item, n=10):
  closest_items_map = dict()
  for id, _ in id_to_item.items():
    distances = distance_matrix[id]
    closest_items = sorted(range(len(distances)), key=lambda k: distances[k])
    closest_items_map[id] = closest_items[:n]
  return closest_items_map

embs = load_emb('GMF', 'user')
# distance_matrix_ = get_distance_matirx(embs)
# id_to_user_map = get_id_to_user_map()
# closest_users_map = get_n_closest_items(distance_matrix_, id_to_user_map)


# print(distance_matrix.shape)
# with open('10_closest_movies.pkl', 'rb') as pkl_file:
#   closest_movies_map = pickle.load(pkl_file)
#
# with open("closest_users_embeddings.txt","w+") as f:
#   for k, v in closest_users_map.items():
#     for mv in v:
#       if mv in id_to_user_map:
#
#         f.write((" ".join(id_to_user_map[mv])))
#     f.write("----\n")


X_embedded = perfTSNE(embs)

plotPoints(X_embedded, 'g.', "all")

# males = get_users({"Gender": "M", "Age": [1, 1]})
males = get_users({"Age": [56, 56]})
horror_embedded_vectors = getIdsEmbeddings(X_embedded, males)
plotPoints(horror_embedded_vectors, 'r.', "males")
#
# # scifi_item_embs = getMovieIds(["Fantasy"])
# # scifi_embedded_vectors = getIdsEmbeddings(X_embedded, scifi_item_embs)
# # plotPoints(scifi_embedded_vectors, 'b.', "scifi")
#
# hns_item_embs = getMovieIds(["Animation"])
# hns_embedded_vectors = getIdsEmbeddings(X_embedded, hns_item_embs)
# plotPoints(hns_embedded_vectors, 'y.', "scifi-horror")
plt.show()

'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None
conf_vs_acc_map = None
def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global conf_vs_acc_map
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    conf_vs_acc_map = {(round(k,1)):[0,0] for k in np.arange(0,1,0.1)}

    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)

    conf_map = {k: v[1] / (v[1] + v[0] + 0.00001)
                for k, v in conf_vs_acc_map.items()}

    print(conf_map)
    import matplotlib.pyplot as plt
    # hr_vs_conf_map = {k:np.count_nonzero(v)/len(v) for k,v in hits_map.items()}
    plt.bar(range(len(conf_map.keys())), list(conf_map.values()), tick_label=list(conf_map.keys()))
    plt.bar(range(len(conf_map.keys())), list(map(lambda x: x + 0.05, list(conf_map.keys()))),
            tick_label=list(conf_map.keys()), color=(0,0,0,0), edgecolor='g')
    # # plt.hist(scores, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.show()
    # exit(0)
    return (hits, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = _model.predict([users, np.array(items)],
                                 batch_size=100, verbose=0)
    # _model.evaluate(x=[np.array([1]), np.array([25])], y=np.array([0]),
    #                 verbose=1)

    # output_func = K.function([_model.layers[0].input, _model.layers[1].input],[_model.layers[6].output, _model.layers[7].output])
    # Note here first argument is the item id and the 2nd is the user id.
    # output_func(inputs=[np.reshape([2791],(1,1)), np.reshape([0],(1,1))])

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i][0]
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)

    expected_argmax = [1] * 99
    expected_argmax.append(0)
    for i in range(0,len(predictions)):
      confidence = np.max(predictions[i])
      if np.argmax(predictions[i]) == expected_argmax[i]:
        conf_vs_acc_map[confidence // 0.1 / 10][1] += 1
      else:
        conf_vs_acc_map[confidence // 0.1 / 10][0] += 1
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

# Non discounted cumulative gain.
def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

import os
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans
import pickle
from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx
from matplotlib import pyplot as plt
import itertools
from scipy.spatial import distance

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    # file_dirs = ["./corpus/文本材料/官方授权清单", "./corpus/文本材料/非官方授权清单"]
    file_dirs = ["./corpus/文本材料-终版/官方授权清单", "./corpus/文本材料-终版/非官方授权清单"]
    lags = ["/zh", "/en"]
    # lags = ["/en"]
    # file_dirs = ["./corpus/文本材料/官方授权清单"]
    # lags = ["/zh"]
    embedding_path = {'/zh': "./data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2",
                      '/en': "./data/GoogleNews-vectors-negative300.bin.gz"}
    binary = {'/zh': False, '/en': True}
    threshold = 0
    for lag in lags:
        word_vectors = KeyedVectors.load_word2vec_format(embedding_path[lag], binary=binary[lag])
        print(lag, ' embedding loaded')
        for file_dir in file_dirs:
            fns = os.listdir(file_dir + lag)
            All_keywords = set()
            corpus_keywords = []
            txt_fns = []
            for fn in fns:
                if fn.find(".txt") < 0:
                    continue
                txt_fns.append(fn)
            for fn in txt_fns:
                # 读入数据
                f = open(file_dir + lag + "/key_words/" + fn[:-4] + "_keywords.txt", "r", encoding="utf8")
                temp_words = f.readlines()
                words = [wd.strip().split("\t\t")[0] for wd in temp_words if wd.strip()]  # 停用词去除
                f.close()
                corpus_keywords.append(words.copy())
                All_keywords = All_keywords.union(words)
            # 词向量

            All_keywords = list(All_keywords)
            valid_keywords = []
            wv = []
            for word in All_keywords:
                if word in word_vectors:
                    wv.append(word_vectors[word])
                    valid_keywords.append(word)
            wv = np.array(wv)
            # 存储
            with open(file_dir+lag+"/word2vec.pkl", 'wb') as f:
                pickle.dump([wv, valid_keywords], f)
            print(file_dir, lag, 'process completed!')
            print("keyword number: ", len(valid_keywords))

            # 读取词向量
            with open(file_dir + lag + "/word2vec.pkl", 'rb') as f:
                wv, valid_keywords = pickle.load(f)
            # 转换为单位向量
            uv = np.linalg.norm(wv, axis=1).reshape(-1, 1)  # unit vector
            wv = wv / uv  # vector or matrix norm

            # 聚类
            labels = KMeans(n_clusters=12).fit(wv).labels_

            # 输出excel
            df = pd.DataFrame([(valid_keywords[e], labels[e]) for e in range(len(valid_keywords))], columns=['word','label'])
            df.sort_values(by='label', inplace=True)
            df.to_excel(file_dir+lag+"/word_cluster.xlsx", index=False)

            """
            # 构建节点集合和边集合
            combs = list(itertools.combinations(valid_keywords, 2))
            edges = []
            dist = distance.cdist(wv, wv, 'cosine')
            dist = 1-(dist/np.max(dist))
            idx1_ls, idx2_ls = np.nonzero(dist > threshold)
            for i in range(len(idx1_ls)):
                idx1 = idx1_ls[i]
                idx2 = idx2_ls[i]
                if idx1 >= idx2:
                    continue
                cos = dist[idx1, idx2]
                # edges.append((valid_keywords[idx1], valid_keywords[idx2],  {'weight': cos / threshold}))
                # edges.append((valid_keywords[idx1], valid_keywords[idx2], {'weight': cos}))
                edges.append((valid_keywords[idx1], valid_keywords[idx2], {'weight': 1}))

            # 构建图
            G = nx.Graph()
            G.add_nodes_from(valid_keywords)
            G.add_edges_from(edges)

            # 社区发现
            c = greedy_modularity_communities(G, n_communities=7)
            # 颜色设置
            color_maps = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "pink"]
            colors = [0] * len(valid_keywords)
            for idx in range(len(valid_keywords)):
                colors[idx] = color_maps[labels[idx]]

            
            for i in range(len(c)):
                for node in c[i]:
                    idx = valid_keywords.index(node)
                    colors[idx] = color_maps[i]
                    G.nodes[node]['subset'] = i
            
            # 绘图
            subax1 = plt.subplot(111)


            pos = nx.kamada_kawai_layout(G)
            # pos = nx.multipartite_layout(G)
            # pos = nx.spectral_layout(G)
            # pos = nx.random_layout(G)
            edgewidth = [G.get_edge_data(u, v)['weight']/100 for u, v in G.edges()]
            # nodesize = [cnt for cnt in nodes_cnt]
            nodesize = 3
            # options = {"node_size": 1500, "alpha": 0.3, "node_color": colors}
            # nx.draw(G, with_labels=True, font_weight='bold', width=edgewidth, **options)
            # nx.draw(G, font_weight='bold', width=edgewidth, node_size=0.1)
            options = {"node_size": nodesize, "alpha": 0.5, "width": edgewidth,
                       "node_color": colors, "edge_color": "darkgrey"}
            nx.draw(G, pos, with_labels=True, font_size=3, **options)
            plt.savefig(file_dir + lag + "/community/semantic-network.png")
            plt.show()
            """
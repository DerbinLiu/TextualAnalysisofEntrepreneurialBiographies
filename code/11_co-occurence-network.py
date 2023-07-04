import os
import numpy as np
import pandas as pd
import itertools
import pickle
from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx
from matplotlib import pyplot as plt
# from communities.algorithms import louvain_method
# from communities.visualization import draw_communities
# from communities.visualization import louvain_animation


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    # file_dirs = ["./corpus/文本材料/官方授权清单", "./corpus/文本材料/非官方授权清单"]
    file_dirs = ["./corpus/文本材料-终版/官方授权清单", "./corpus/文本材料-终版/非官方授权清单"]
    lags = ["/zh", "/en"]
    # lags = ["/zh"]
    # lags = ["/en"]
    # file_dirs = ["./corpus/文本材料/官方授权清单"]
    # lags = ["/zh"]

    for file_dir in file_dirs:
        for lag in lags:

            fns = os.listdir(file_dir + lag)
            # fns = ["腾讯传.txt"]

            # All_keywords = set()
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
                words = [wd.strip().split("\t\t")[0].lower() for wd in temp_words if wd.strip()]  # 停用词去除
                f.close()
                corpus_keywords.append(words.copy())
                # All_keywords = All_keywords.union(words)
            # 构建词汇表
            with open(file_dir + lag + "/word2vec.pkl", 'rb') as f:
                wv, valid_keywords = pickle.load(f)

            nodes = list(valid_keywords)
            nodes_cnt = [0]*len(nodes)
            for i in range(len(nodes)):
                for keywords in corpus_keywords:
                    if nodes[i] in keywords:
                        nodes_cnt[i] += 1
            print(len(nodes))
            # 构建共现矩阵
            edge_weights = {}

            # combs = list(itertools.combinations(nodes, 2))
            combs = list(itertools.combinations(list(range(len(nodes))), 2))
            for comb in combs:
                for keywords in corpus_keywords:
                    if nodes[comb[0]] in keywords and nodes[comb[1]] in keywords:
                        if comb in edge_weights.keys():
                            edge_weights[comb] += 1
                        else:
                            edge_weights[comb] = 1
            edges = []
            for key in edge_weights.keys():
                edges.append((nodes[key[0]], nodes[key[1]], {'weight': edge_weights[key]}))
            print("edges number: ", len(edges))
            # 存储
            with open(file_dir+lag+'/community/edges.txt', 'w', encoding="utf8") as f:
                f.write("source, target, weight\n")
                for edge in edges:
                    f.write("%s, %s, %d\n" % (edge[0], edge[1], edge[2]['weight']))
            with open(file_dir+lag+"/community/nodes_edges.pkl", "wb") as f:
                pickle.dump([nodes, edges, nodes_cnt], f)

            """
            # 读取
            with open(file_dir+lag+"/community/nodes_edges.pkl", "rb") as f:
                nodes, edges, nodes_cnt = pickle.load(f)

            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            # 社区发现
            c = greedy_modularity_communities(G, n_communities=4)
            # 绘图
            subax1 = plt.subplot(111)
            color_maps = ["red", "orange", "yellow", "green", "aqua", "blue", "pink"]
            colors = [0] * len(nodes)
            for i in range(len(c)):
                for node in c[i]:
                    idx = nodes.index(node)
                    colors[idx] = color_maps[i]
                    G.nodes[node]['subset'] = i

            # pos = nx.kamada_kawai_layout(G)
            pos = nx.multipartite_layout(G)
            # pos = nx.spectral_layout(G)
            # pos = nx.random_layout(G)
            edgewidth = [G.get_edge_data(u, v)['weight']/50. for u, v in G.edges()]
            nodesize = [cnt for cnt in nodes_cnt]
            # nodesize = 3
            #options = {"node_size": 1500, "alpha": 0.3, "node_color": colors}
            # nx.draw(G, with_labels=True, font_weight='bold', width=edgewidth, **options)
            # nx.draw(G, font_weight='bold', width=edgewidth, node_size=0.1)
            options = {"node_size": nodesize, "alpha": 0.5, "width": edgewidth,
                       "node_color": colors, "edge_color": "darkgrey"}
            nx.draw(G, pos, with_labels=True, font_size=3, **options)
            plt.savefig(file_dir+lag+"/community/co-occurence-network.png")
            plt.show()
            """

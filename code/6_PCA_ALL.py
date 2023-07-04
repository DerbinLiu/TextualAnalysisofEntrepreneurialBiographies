import pickle
import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    """
    lw = 10000
    P = 100
    # 读取情感词典
    LabMT = {}
    Havg_list = []
    for line in open("./LabMT/labMT2chinese-backup.txt", "r", encoding="utf8"):
        word_list = line.split()
        if len(word_list) >= 2:
            LabMT[word_list[0]] = eval(word_list[2])
            Havg_list.append(eval(word_list[2]))

    for line in open("./LabMT/labMT2english-backup.txt", "r", encoding="utf8"):
        word_list = line.split()
        if len(word_list) >= 2:
            LabMT[word_list[0]] = eval(word_list[2])
            Havg_list.append(eval(word_list[2]))
    Havg = np.mean(Havg_list)

    hownet = {}
    dict_f = open("./hownet/pos.pkl", "rb")
    hownet["pos"] = pickle.load(dict_f)
    dict_f = open("./hownet/neg.pkl", "rb")
    hownet["neg"] = pickle.load(dict_f)
    pos_score = []
    pos_wd = []
    neg_score = []
    neg_wd = []
    for word in hownet["pos"]:
        if word in LabMT.keys():
            pos_score.append(LabMT[word])
            pos_wd.append(word)
    for word in hownet["neg"]:
        if word in LabMT.keys():
            neg_score.append(LabMT[word])
            neg_wd.append(word)
    pos_mean = np.mean(pos_score)
    print("pos min: ", np.min(pos_score), " max: ", np.max(pos_score), " mean: ", pos_mean)
    neg_mean = np.mean(neg_score)
    print("neg min: ", np.min(neg_score), " max: ", np.max(neg_score), " mean: ", neg_mean)
    """

    # file_dirs = ["./corpus/文本材料/官方授权清单", "./corpus/文本材料/非官方授权清单"]
    file_dirs = ["./corpus/文本材料-终版/官方授权清单", "./corpus/文本材料-终版/非官方授权清单"]
    lags = ["/zh", "/en"]
    # combs = ["/zh", "/en", "/all"]
    combs = ["/zh", "/en", "/all", "/官方", "/非官"]
    comp_num = 4
    top_num = 3

    All_scores = []
    All_fns = []
    for file_dir in file_dirs:
        for lag in lags:
            with open(file_dir+lag+"/avg_score.pkl", "rb") as f:
                data = pickle.load(f)
                All_scores.append(data[0])
                All_fns.append(data[2])
    # 0: 官方中文
    # 1: 官方英文
    # 2: 非官中文
    # 3: 非官英文
    scores = {}
    fns = {}
    # 中文
    scores["/zh"] = All_scores[0].copy()
    scores["/zh"].extend(All_scores[2].copy())
    fns["/zh"] = All_fns[0].copy()
    fns["/zh"].extend(All_fns[2].copy())
    # 英文
    scores["/en"] = All_scores[1].copy()
    scores["/en"].extend(All_scores[3].copy())
    fns["/en"] = All_fns[1].copy()
    fns["/en"].extend(All_fns[3].copy())
    # 全部
    scores["/all"] = All_scores[0].copy()
    scores["/all"].extend(All_scores[1].copy())
    scores["/all"].extend(All_scores[2].copy())
    scores["/all"].extend(All_scores[3].copy())
    fns["/all"] = All_fns[0].copy()
    fns["/all"].extend(All_fns[1].copy())
    fns["/all"].extend(All_fns[2].copy())
    fns["/all"].extend(All_fns[3].copy())
    # 官方
    scores["/官方"] = All_scores[0].copy()
    scores["/官方"].extend(All_scores[1].copy())
    fns["/官方"] = All_fns[0].copy()
    fns["/官方"].extend(All_fns[1].copy())
    # 非官
    scores["/非官"] = All_scores[2].copy()
    scores["/非官"].extend(All_scores[3].copy())
    fns["/非官"] = All_fns[2].copy()
    fns["/非官"].extend(All_fns[3].copy())

    for comb in combs:
        scores_array = np.array(scores[comb].copy())
        fns_ls = fns[comb]
        # All_scores_array_zeros = StandardScaler.fit_transform(All_scores_array)
        # All_scores_array_zeros = All_scores_array - np.mean(All_scores_array, axis=0)
        scores_array_zeros = (scores_array.T - np.mean(scores_array, axis=1)).T
        U, sigma, VT = np.linalg.svd(scores_array_zeros)
        p_len = VT.shape[0]
        variance_ratio = sigma ** 2 / np.sum(sigma ** 2)
        print("VT:", np.min(VT[:4, :]), np.max(VT[:4, :]))

        color_maps = ["#607588", "#BA9048", "#E0BA87", "#998F7C"]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(range(p_len), VT[0, :], color_maps[0])
        ax.plot(range(p_len), VT[1, :], color_maps[1])
        ax.plot(range(p_len), VT[2, :], color_maps[2])
        ax.plot(range(p_len), VT[3, :], color_maps[3])
        # plt.plot(range(p_len), VT[4, :], "#")
        # plt.plot(range(p_len), VT[5, :], "m")
        ax.legend(["comp 1: %.2f" % variance_ratio[0],
                   "comp 2: %.2f" % variance_ratio[1],
                   "comp 3: %.2f" % variance_ratio[2],
                   "comp 4: %.2f" % variance_ratio[3]], loc="upper left")
        """
        plt.legend(["comp 1: %.2f" % variance_ratio[0],
                    "comp 2: %.2f" % variance_ratio[1],
                    "comp 3: %.2f" % variance_ratio[2],
                    "comp 4: %.2f" % variance_ratio[3],
                    "comp 5: %.2f" % variance_ratio[4],
                    "comp 6: %.2f" % variance_ratio[5]])
        """
        ax.set_ylim(-0.30, 0.30)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
        plt.savefig("./corpus/文本材料-终版/svd"+comb+"/all_pca.jpg")
        plt.close()
        for i in range(4):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(range(p_len), VT[i, :], color_maps[i])
            # plt.legend(["comp %d: %.2f" % (i, variance_ratio[i])])
            ax.set_title("comp " + str(i+1))
            ax.set_ylim(-0.30, 0.30)
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
            plt.savefig("./corpus/文本材料-终版/svd"+comb+"/all_pca_comp-"+str(i)+".jpg")
            plt.close()


        book_idx = []
        av_max = -10
        av_min = 10
        for i in range(comp_num):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            sv = VT[i, :]  # 第i个主成分
            W = []
            tmp_idx = []
            for j in range(len(fns_ls)):
                W.append((j, sigma[i]*U[j, i]))
            nw_W = sorted(W, key=lambda x: x[1], reverse=True)
            # 选择前 top_num 本书
            ax.plot(range(p_len), sv, "r", linewidth=2)
            for k in range(top_num):
                idx = nw_W[k][0]
                weight = nw_W[k][1]
                if weight <= 0:
                    break
                else:
                    tmp_idx.append(idx)
                    y = scores_array_zeros[idx, :]/abs(weight)
                    av_min = min(av_min, np.min(y))
                    av_max = max(av_max, np.max(y))
                    ax.plot(range(p_len), y, "darkgrey", linewidth=0.5)

            ax.legend(["SV "+str(i+1), "closest "+str(top_num)+" books"], loc="upper right")
            ax.set_ylim(-0.50, 0.50)
            # ax.set_ylim(-1.00, 0.90)
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
            plt.savefig("./corpus/文本材料-终版/svd"+comb+"/sv_"+str(i+1)+".jpg")
            plt.close()
            book_idx.append(tmp_idx.copy())

        for i in range(comp_num):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            sv = VT[i, :]  # 第i个主成分
            W = []
            tmp_idx = []
            for j in range(len(fns_ls)):
                W.append((j, sigma[i]*U[j, i]))
            nw_W = sorted(W, key=lambda x: x[1])
            # 选择前 top_num 本书
            ax.plot(range(p_len), -sv, "r", linewidth=2)
            for k in range(top_num):
                idx = nw_W[k][0]
                weight = nw_W[k][1]
                if weight > 0:
                    break
                else:
                    tmp_idx.append(idx)
                    y = scores_array_zeros[idx, :] / abs(weight)
                    av_min = min(av_min, np.min(y))
                    av_max = max(av_max, np.max(y))
                    ax.plot(range(p_len), y, "darkgrey", linewidth=0.5)
            ax.legend(["-(SV " + str(i+1)+")", "closest "+str(top_num)+" books"], loc="upper right")
            ax.set_ylim(-0.50, 0.50)
            # ax.set_ylim(-1.00, 0.90)
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
            plt.savefig("./corpus/文本材料-终版/svd"+comb+"/neg_sv_"+str(i+1)+".jpg")
            plt.close()
            book_idx.append(tmp_idx.copy())

        print("av", av_min, av_max)

        with open("./corpus/文本材料-终版/svd"+comb+"/closest_books.txt", "w", encoding="utf8") as f:
            for i in range(comp_num):
                f.write(" SV "+str(i+1)+"\n")
                for idx in book_idx[i]:
                    f.write(fns_ls[idx]+"\n")
                f.write("-------------------------------\n")
            for i in range(comp_num):
                f.write("neg SV "+str(i+1)+"\n")
                for idx in book_idx[i+comp_num]:
                    f.write(fns_ls[idx]+"\n")
                f.write("-------------------------------\n")

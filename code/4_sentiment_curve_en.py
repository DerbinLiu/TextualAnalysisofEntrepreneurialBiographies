import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    # lw = 10000
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

    # file_dirs = ["./corpus/文本材料/官方授权清单", "./corpus/文本材料/非官方授权清单"]
    file_dirs = ["./corpus/文本材料-终版/官方授权清单", "./corpus/文本材料-终版/非官方授权清单"]
    # file_dirs = ["./corpus/文本材料/对比"]
    for file_dir in file_dirs:
        fns = os.listdir(file_dir + "/en")
        txt_fns = []
        for fn in fns:
            if fn.find(".txt") < 0:
                continue
            txt_fns.append(fn)
        all_scores = []
        all_percentage = []
        mean_scores = []

        # fns = ["Harry Potter and the Deathly Hallows (Book 7) by J. K. Rowling (z-lib.org)"]
        for fn in txt_fns:
            # 读取分词结果
            f = open(file_dir+"/en/cut_text/"+fn[:-4]+"-cut.txt", "r", encoding="utf8")
            words = f.readlines()
            f.close()

            scores = []
            for word in words:
                if word.strip() in LabMT.keys():
                    sc = LabMT[word.strip()]
                else:
                    # sc = Havg
                    sc = -1  # -1表示词典中没有的词
                scores.append(sc)
            with open(file_dir+"/en/cut_score/"+fn[:-4]+"-score.txt", "w", encoding="utf8") as f:
                for i in range(len(scores)):
                    f.write(words[i].strip()+"\t"+str(scores[i])+"\n")

            score_array = np.array(scores)
            N = len(score_array)
            T = int((N-lw)/(P-1))

            avg_score = []
            percentage = []
            mean_scores.append(np.mean(score_array[score_array >= 0]))

            for i in range(P):
                start_pos = i*T
                sub_array = score_array[start_pos:start_pos+lw]
                avg_score.append(np.mean(sub_array[sub_array >= 0]))
                percentage.append((start_pos + lw / 2) / N)
            percentage = np.array(percentage)

            # plt.plot(percentage*100, avg_score, color="#E8D3BC")
            # plt.xlim(0, 100)
            # plt.savefig(file_dir+"/en/fig/"+fn[:-4]+".jpg")
            # plt.close()
            all_scores.append(avg_score.copy())
            all_percentage.append(percentage.copy())
        with open(file_dir+"/en/avg_score.pkl", "wb") as f:
            pickle.dump([all_scores, all_percentage, txt_fns, mean_scores], f)
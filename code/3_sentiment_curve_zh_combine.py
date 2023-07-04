import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
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

    # file_dirs = ["./corpus/文本材料/官方授权清单", "./corpus/文本材料/非官方授权清单"]
    # file_dirs = ["./corpus/文本材料/小说"]
    # file_dirs = ["./corpus/文本材料/对比"]
    file_dirs = ["./corpus/文本材料-终版/官方授权清单", "./corpus/文本材料-终版/非官方授权清单"]
    # file_dirs = ["./corpus/文本材料"]
    for file_dir in file_dirs:
        fns = os.listdir(file_dir + "/zh")
        txt_fns = []
        for fn in fns:
            if fn.find(".txt") < 0:
                continue
            txt_fns.append(fn)
        all_scores = []
        all_percentage = []
        mean_scores = []

        for fn in txt_fns:
            # 读取分词结果
            f = open(file_dir+"/zh/cut_text/"+fn[:-4]+"-cut.txt", "r", encoding="utf8")
            temp_words = f.readlines()
            f.close()
            words = [wd.strip().lower() for wd in temp_words]

            scores = []
            for wd in words:
                if wd in LabMT.keys():
                    sc = LabMT[wd]
                else:
                    sc = -1  # -1表示词典中没有的词
                scores.append(sc)
            pos_bound = 6
            neg_bound = 4
            for i in range(len(words)):
                if scores[i] < 0:
                    word = words[i]
                    if word in hownet["pos"]:
                        prev_ = -1
                        next_ = -1
                        begin = max(i-1, 0)
                        end = max(0, i-1001)
                        for j in range(begin, end, -1):
                            if scores[j] >= pos_bound:
                                prev_ = scores[j]
                                break
                        begin = min(i+1, len(words))
                        end = min(i+10001, len(words))
                        for j in range(begin, end):
                            if scores[j] >= pos_bound:
                                next_ = scores[j]
                                break
                        if prev_ >= 0 and next_ >= 0:
                            scores[i] = (prev_ + next_)/2
                        elif prev_ >= 0:
                            scores[i] = prev_
                        elif next_ >= 0:
                            scores[i] = next_
                    elif word in hownet["neg"]:
                        prev_ = -1
                        next_ = -1
                        begin = max(i-1, 0)
                        end = max(i-1001, 0)
                        for j in range(begin, end, -1):
                            if 0 <= scores[j] <= neg_bound:
                                prev_ = scores[j]
                                break
                        begin = min(i+1, len(words))
                        end = min(i+1, len(words))
                        for j in range(begin, end):
                            if 0 <= scores[j] <= neg_bound:
                                next_ = scores[j]
                                break
                        if prev_ >= 0 and next_ >= 0:
                            scores[i] = (prev_ + next_) / 2
                        elif prev_ >= 0:
                            scores[i] = prev_
                        elif next_ >= 0:
                            scores[i] = next_
                    else:
                        pass

            with open(file_dir+"/zh/cut_score/"+fn[:-4]+"-score.txt", "w", encoding="utf8") as f:
                for i in range(len(scores)):
                    f.write(words[i]+"\t" + str(scores[i]) + "\n")

            score_array = np.array(scores)
            N = len(score_array)

            avg_score = []
            percentage = []
            mean_scores.append(np.mean(score_array[score_array >= 0]))
            # 开头算3个点
            """
            T = 1250
            for i in range(3):
                length = 2500
                start_pos = i * T
                sub_array = score_array[start_pos:start_pos+length]
                avg_score.append(np.mean(sub_array[sub_array >= 0]))
                percentage.append((start_pos+length/2)/N)
            """
            # 中间正常算
            T = int((N - lw) / (P - 1))
            for i in range(P):
                start_pos = i * T
                sub_array = score_array[start_pos:start_pos + lw]
                avg_score.append(np.mean(sub_array[sub_array >= 0]))
                percentage.append((start_pos+lw/2)/N)
            # 结尾加3个点
            """
            T = 1250
            start_pos = N - 5000
            for i in range(3):
                length = T*2
                sub_array = score_array[start_pos:start_pos+length]
                avg_score.append(np.mean(sub_array[sub_array >= 0]))
                percentage.append((start_pos+length/2)/N)
                start_pos += T
            """
            percentage = np.array(percentage)
            # plt.plot(percentage*100, avg_score, color="#E8D3BC")
            # plt.xlim(0, 100)
            # plt.savefig(file_dir+"/zh/fig/"+fn[:-4]+".jpg")
            # plt.close()
            all_scores.append(avg_score.copy())
            all_percentage.append(percentage.copy())

        with open(file_dir+"/zh/avg_score.pkl", "wb") as f:
            pickle.dump([all_scores, all_percentage, txt_fns, mean_scores], f)

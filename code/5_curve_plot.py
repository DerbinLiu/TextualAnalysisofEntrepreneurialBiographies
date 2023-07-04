import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import csv

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    # file_dirs = ["./corpus/文本材料/官方授权清单", "./corpus/文本材料/非官方授权清单"]
    # lags = ["/zh", "/en"]
    # file_dirs = ["./corpus/文本材料/对比"]
    # lags = ["/zh"]
    file_dirs = ["./corpus/文本材料-终版/官方授权清单", "./corpus/文本材料-终版/非官方授权清单"]
    lags = ["/zh", "/en"]

    # file_dirs = ["./corpus/文本材料"]
    # file_dirs = ["./corpus/文本材料/小说"]

    # lags = ["/zh"]
    stats = []
    avg_min = {"/zh": 5.08, "/en": 5.19}
    avg_max = {"/zh": 5.42, "/en": 5.62}
    for file_dir in file_dirs:
        text_type = file_dir.strip("./corpus/文本材料-终版/").strip("授权清单")
        for lag in lags:
            with open(file_dir+lag+"/avg_score.pkl", "rb") as f:
                data = pickle.load(f)
                avg_scores = data[0]
                percentages = data[1]
                fns = data[2]
                mean_scores = data[3]
            # avg_min = np.min(avg_scores)
            # avg_max = np.max(avg_scores)
            print(file_dir, lag, np.min(avg_scores), np.max(avg_scores))

            for i in range(len(avg_scores)):
                # 做一些统计计算
                curve_mean = np.mean(avg_scores[i])
                stat = []
                stat.append(fns[i].replace(",", "-"))  # 书名
                stat.append(mean_scores[i])  # 全书均值
                stat.append(curve_mean)  # 曲线均值
                min_sc = np.min(avg_scores[i])  # 最小值
                max_sc = np.max(avg_scores[i])  # 最大值
                stat.append(min_sc)  # 最小值
                stat.append(max_sc)  # 最大值
                stat.append(max_sc-min_sc)  # 极差
                stat.append(np.var(avg_scores[i]))  # 方差
                stat.append(np.std(avg_scores[i]))  # 标准差
                stat.append(lag.strip("/"))  # 中英文
                stat.append(text_type)  # 是否官方

                stats.append(stat.copy())  # 汇总


                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(percentages[i]*100, avg_scores[i], color="#DECEA6")
                ax.plot(percentages[i]*100, np.ones(100)*curve_mean, '--', dashes=(5, 5), color="#C8C8C8", linewidth=0.75)
                ax.set_xlim(0, 100)
                ax.set_ylim(avg_min[lag], avg_max[lag])
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
                plt.savefig(file_dir + lag + "/fig/" + fns[i][:-4] + ".jpg")
                plt.close()

            # 写csv

            with open(file_dir+lag+"/avg_score_"+lag.strip("/")+"_"+text_type+".csv", "w", encoding="utf8", newline="") as f:
                csv_writer = csv.writer(f)
                for i in range(len(avg_scores)):
                    row = []
                    row.append(fns[i].replace(",", "-"))
                    row.extend(avg_scores[i])
                    csv_writer.writerow(row)

    # 写txt
    with open("./corpus/文本材料-终版/统计值.txt", "w", encoding="utf8") as f:
        f.write("书名, 全书均值, 曲线均值, 最小值, 最大值, 极差, 方差, 标准差, 语言, 是否官方\n")
        for stat in stats:
            f.write("%s, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %s, %s\n" % tuple(stat))

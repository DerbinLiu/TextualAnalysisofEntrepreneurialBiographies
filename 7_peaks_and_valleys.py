import pickle
import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle
import scipy.signal as signal

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    # file_dirs = ["./corpus/文本材料/官方授权清单", "./corpus/文本材料/非官方授权清单"]
    file_dirs = ["./corpus/文本材料-终版/官方授权清单", "./corpus/文本材料-终版/非官方授权清单"]
    lags = ["/zh", "/en"]
    # file_dirs = ["./corpus/文本材料/官方授权清单"]
    # lags = ["/zh"]
    n = 5
    offset = int(n/2)
    sep = 1
    all_peaks_valleys = []
    for file_dir in file_dirs:
        text_type = file_dir.strip("./corpus/文本材料-终版/").strip("授权清单")
        for lag in lags:
            with open(file_dir + lag + "/avg_score.pkl", "rb") as f:
                data = pickle.load(f)
                avg_scores = data[0]
                percentages = data[1]
                fns = data[2]
                mean_scores = data[3]

            N = len(avg_scores)
            for i in range(N):
                # 计算每一条曲线中波峰波谷的数量
                avg_score = avg_scores[i]
                curve_mean = np.mean(avg_score)
                curve_std = np.std(avg_score)
                # new_score = np.convolve(avg_score, np.ones(n, )/n, mode='valid')
                # indexs = range(offset, len(new_score) + offset)
                curve_len = len(avg_score)
                new_score = [avg_score[sep*i] for i in range(int(curve_len/sep)+1) if sep*i < curve_len]
                new_score = np.array(new_score)
                indexs = [sep*i for i in range(int(curve_len/sep)+1) if sep*i < curve_len]

                # 作图
                """
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(percentages[i]*100, avg_score, color='red')
                ax.plot(percentages[i][indexs] * 100, new_score, color='blue')
                ax.plot(percentages[i] * 100, np.ones(100) * curve_mean, '--', dashes=(5, 5), color="#C8C8C8",
                        linewidth=0.75)
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
                ax.set_xlim(0, 100)
                ax.legend(["original curve", "adjusted curve"], loc="upper left")
                plt.savefig(file_dir + lag + "/peaks_valleys/" + fns[i][:-4] + ".jpg")
                plt.close()
                # plt.show()
                """

                # 统计波峰数量
                peak_valley_cnt = []
                peak_valley_cnt.append(fns[i].replace(",", "-"))
                peak_valley_cnt.append(lag.strip("/"))  # 中英文
                peak_valley_cnt.append(text_type)  # 是否官方

                peak_valley_cnt.append(len(signal.find_peaks(new_score)[0]))  # 波峰
                peak_valley_cnt.append(len(signal.find_peaks(-new_score)[0]))  # 波谷

                peak_valley_cnt.append(len(signal.find_peaks(new_score, curve_mean + curve_std)[0]))  # 波峰
                peak_valley_cnt.append(len(signal.find_peaks(new_score, curve_mean + 2 * curve_std)[0]))  # 波峰
                peak_valley_cnt.append(len(signal.find_peaks(new_score, curve_mean + 3 * curve_std)[0]))  # 波峰

                peak_valley_cnt.append(len(signal.find_peaks(-new_score, -curve_mean + curve_std)[0]))  # 波谷
                peak_valley_cnt.append(len(signal.find_peaks(-new_score, -curve_mean + 2 * curve_std)[0]))  # 波谷
                peak_valley_cnt.append(len(signal.find_peaks(-new_score, -curve_mean + 3 * curve_std)[0]))  # 波谷

                peak_valley_cnt.append(len(signal.find_peaks(-new_score, height=[-100, -curve_mean])[0]))  # 波谷
                # peak_valley_cnt.append(len(signal.find_peaks(-new_score, height=[-curve_mean - 2 * curve_std, -curve_mean])[0]))  # 波谷
                # peak_valley_cnt.append(len(signal.find_peaks(-new_score, height=[-curve_mean - 3 * curve_std, -curve_mean])[0]))  # 波谷

                peak_valley_cnt.append(len(signal.find_peaks(new_score, height=[-100, curve_mean])[0]))  # 波峰
                # peak_valley_cnt.append(len(signal.find_peaks(new_score, height=[curve_mean - 2 * curve_std, curve_mean])[0]))  # 波峰
                # peak_valley_cnt.append(len(signal.find_peaks(new_score, height=[curve_mean - 3 * curve_std, curve_mean])[0]))  # 波峰

                all_peaks_valleys.append(peak_valley_cnt.copy())

    # 写txt
    with open("./corpus/文本材料-终版/波峰波谷统计-1.txt", "w", encoding="utf8") as f:
        f.write("书名, 语言, 是否官方, peaks, valleys, "
                "peaks_(mu+std&infinity),  peaks_(mu+2std&infinity),  peaks_(mu+3std&infinity),"
                "valleys_(-infinity&mu-std), valleys_(-infinity&mu-2std), valleys_(-infinity&mu-3std), "
                "valleys_(mu&infinity), "
                "peaks_(-infinity&mu)\n")
        for peak_valley in all_peaks_valleys:
            f.write("%s, %s, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(peak_valley))

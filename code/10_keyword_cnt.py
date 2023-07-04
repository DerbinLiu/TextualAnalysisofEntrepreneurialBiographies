import os
## 关键词数量统计

if __name__ == "__main__":
    # file_dirs = ["./corpus/文本材料/官方授权清单", "./corpus/文本材料/非官方授权清单"]
    file_dirs = ["./corpus/文本材料-终版/官方授权清单", "./corpus/文本材料-终版/非官方授权清单"]
    lags = ["/zh", "/en"]
    # lags = ["/en"]
    # file_dirs = ["./corpus/文本材料/官方授权清单"]
    # lags = ["/zh"]
    for file_dir in file_dirs:
        for lag in lags:
            fns = os.listdir(file_dir + lag)  # 获取文件名
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

            All_keywords = list(All_keywords)  # 关键词列表

            All_keyword_cnt = []
            Total_cnt = [0] * len(All_keywords)
            Total_occur = [0] * len(All_keywords)
            for fn in txt_fns:
                # 读入每一本书的分词文件
                book_word_freq = {}
                with open(file_dir+lag+"/word_frequence/"+fn[:-4]+"-word_freq.txt", 'r', encoding="utf8") as f:
                    lines = f.readlines()
                    for line in lines:
                        line_ls = line.strip().split("\t\t")
                        book_word_freq[line_ls[0]] = line_ls[1]
                # 统计每一本书中关键词的数量
                keyword_cnt = []
                for i in range(len(All_keywords)):
                    wd = All_keywords[i]
                    if wd in book_word_freq.keys():
                        freq = book_word_freq[wd]
                        keyword_cnt.append(freq)
                        Total_cnt[i] += eval(freq)
                        Total_occur[i] += 1
                    else:
                        keyword_cnt.append("0")
                All_keyword_cnt.append(keyword_cnt.copy())
            # 输出到文件
            with open(file_dir+lag+"/key_words/keywords_cnt.csv", "w", encoding="utf8") as f:
                f.write("书名, " + ", ".join(All_keywords)+"\n")
                for i in range(len(txt_fns)):
                    fn = txt_fns[i]
                    f.write(fn[:-4].replace(",", "-")+", "+", ".join(All_keyword_cnt[i])+"\n")
                f.write("ALL, " + ", ".join([str(freq) for freq in Total_cnt])+"\n")
                f.write("出现次数, " + ", ".join([str(occur) for occur in Total_occur])+"\n")

import os
import jieba.analyse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import re


if __name__ == "__main__":
    # 词频统计与分析
    # file_dirs = ["./corpus/文本材料/官方授权清单", "./corpus/文本材料/非官方授权清单"]
    file_dirs = ["./corpus/文本材料-终版/官方授权清单", "./corpus/文本材料-终版/非官方授权清单"]
    lags = ["/zh", "/en"]
    # lags = ["/en"]
    # file_dirs = ["./corpus/文本材料/官方授权清单"]
    # lags = ["/zh"]

    # 导入中文停用词
    f = open("./中文停用词/百度停用词列表.txt", "r", encoding="gb18030")
    temp_words = f.readlines()
    f.close()
    stopwords = [wd.strip() for wd in temp_words]

    # 导入英文停用词
    f = open("./英文停用词/english_stopwords.txt", "r", encoding="utf8")
    temp_words = f.readlines()
    f.close()
    en_stopwords = [wd.strip() for wd in temp_words]

    stopwords.extend(en_stopwords)
    # 自定义停用词
    f = open("./中文停用词/自定义停用词.txt", "r", encoding="utf8")
    temp_words = f.readlines()
    f.close()
    temp_stopwords = [wd.strip() for wd in temp_words]
    stopwords.extend(temp_stopwords)
    # 手动添加部分停用词
    stopwords.extend(["Sir"])

    # threshhold = 0.0025
    threshold = 0.03
    keyword_num = 5
    for lag in lags:
        corpus = []
        txt_fns = {}
        for file_dir in file_dirs:
            fns = os.listdir(file_dir+lag)
            # fns = ["腾讯传.txt"]
            txt_fns[file_dir] = []
            for fn in fns:
                if fn.find(".txt") < 0:
                    continue
                txt_fns[file_dir].append(fn)

            for fn in txt_fns[file_dir]:
                # 读取分词结果
                f = open(file_dir + lag + "/cut_text/" + fn[:-4] + "-cut.txt", "r", encoding="utf8")
                temp_words = f.readlines()
                # words = [wd.strip() for wd in temp_words if wd.strip() not in stopwords]  # 停用词去除
                words = [wd.strip().lower() for wd in temp_words]
                f.close()

                word_num = len(words)
                word_freq = {}
                for wd in words:
                    if wd not in word_freq.keys():
                        word_freq[wd] = 1
                    else:
                        word_freq[wd] += 1
                corpus.append(" ".join(words.copy()))  # 将文章分词以空格分开，以便符合sklearn数据格式要求
                sorted_dict = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                with open(file_dir + lag + "/word_frequence/" + fn[:-4] + "-word_freq.txt", "w", encoding="utf8") as f:
                    for item in sorted_dict:
                        f.write(item[0] + "\t\t" + str(item[1]) + "\t\t" + str(item[1]/word_num)+"\n")

        # 提取TF
        countVectorizer = CountVectorizer(min_df=0, stop_words=stopwords)
        textVector = countVectorizer.fit_transform(corpus)
        # print(textVector.todense())
        # 计算TF-IDF
        transfomer = TfidfTransformer()
        tfidf = transfomer.fit_transform(textVector).toarray()  # 为词频矩阵的每个词加上权重(即TF * IDF), 得到TF-IDF矩阵
        # print(tfidf)

        # 提取关键词
        # sort = np.argsort(tfidf, axis=1)[:, -keyword_num:]  # 将二维数据中每一行按升序排序，并提取每一行的最后五个（即数值最大的五个）
        # names = countVectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        names = countVectorizer.get_feature_names_out()
        # keywords = pd.Index(names)[sort]
        # keyword_idx = tfidf.toarray >= threshhold
        # keywords = pd.Index(names)[keyword_idx].values

        i = 0
        for file_dir in file_dirs:
            for j in range(len(txt_fns[file_dir])):
                # words = corpus[i]
                # print(keyword_text)

                idxs = np.nonzero(tfidf[i, :] > threshold)[0]
                with open(file_dir+lag+"/key_words/"+txt_fns[file_dir][j][:-4]+"_keywords.txt", "w", encoding="utf8") as f:
                    for idx in idxs:
                        # if tag[1] < threshhold:
                        #    break
                        f.write("%s\t\t %f\n" % (names[idx], tfidf[i, idx]))
                i += 1
            # i += len(txt_fns[file_dir])

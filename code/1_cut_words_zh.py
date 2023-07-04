import csv
import re
import jieba
import numpy as np
import matplotlib.pyplot as plt
import os


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

punc = "~*·！？｡。＂＃＄％＆＇（）()＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

if __name__ == "__main__":
    # 官方授权清单
    # file_dirs = ["./corpus/文本材料", "./corpus/文本材料/官方授权清单", "./corpus/文本材料/非官方授权清单"]
    # file_dirs = ["./corpus/文本材料"]
    # file_dirs = ["./corpus/文本材料/小说"]
    # file_dirs = ["./corpus/文本材料/对比"]
    file_dirs = ["./corpus/文本材料-终版/官方授权清单", "./corpus/文本材料-终版/非官方授权清单"]

    for file_dir in file_dirs:
        fns = os.listdir(file_dir+"/zh")

        for fn in fns:
            if fn.find(".txt") < 0:
                continue
            book = []
            try:
                for line in open(file_dir+"/zh/"+fn, "r", encoding="utf8"):
                    if line != "\n":
                        book.append(re.sub(r"[%s]+" % punc, " ", line))
            except:
                for line in open(file_dir+"/zh/"+fn, "r", encoding="gb18030"):
                    if line != "\n":
                        book.append(re.sub(r"[%s]+" % punc, " ", line))

            with open(file_dir+"/zh/cut_text/"+fn[:-4]+"-cut.txt", "w", encoding="utf8") as f:
                for line in book:
                    seg_list = jieba.cut(line)
                    new_list = [word for word in seg_list if not word.isspace()]
                    f.write('\n'.join(new_list)+"\n")

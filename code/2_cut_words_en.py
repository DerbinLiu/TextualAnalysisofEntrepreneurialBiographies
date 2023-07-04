import nltk, sys, re, collections
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

# patterns that used to find or/and replace particular chars or words
# to find chars that are not a letter, a blank or a quotation
pat_letter = re.compile(r'[^a-zA-Z \']+')
# to find the 's following the pronouns. re.I is refers to ignore case
pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
# to find the 's following the letters
pat_s = re.compile("(?<=[a-zA-Z])\'s")
# to find the ' following the words ending by s
pat_s2 = re.compile("(?<=s)\'s?")
# to find the abbreviation of not
pat_not = re.compile("(?<=[a-zA-Z])n\'t")
# to find the abbreviation of would
pat_would = re.compile("(?<=[a-zA-Z])\'d")
# to find the abbreviation of will
pat_will = re.compile("(?<=[a-zA-Z])\'ll")
# to find the abbreviation of am
pat_am = re.compile("(?<=[I|i])\'m")
# to find the abbreviation of are
pat_are = re.compile("(?<=[a-zA-Z])\'re")
# to find the abbreviation of have
pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

lmtzr = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''


def replace_abbreviations(text):
    new_text = text
    new_text = pat_letter.sub(' ', text).strip().lower()
    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text


# 词形还原
def merge(words):
    new_words = []
    for word in words:
        if word:
            tag = nltk.pos_tag(word_tokenize(word)) # tag is like [('bigger', 'JJR')]
            pos = get_wordnet_pos(tag[0][1])
            if pos:
                lemmatized_word = lmtzr.lemmatize(word, pos)
                new_words.append(lemmatized_word)
            else:
                new_words.append(word)
    return new_words


if __name__ == "__main__":
    # file_dirs = ["./corpus/文本材料/官方授权清单", "./corpus/文本材料/非官方授权清单"]
    # file_dirs = ["./corpus/文本材料/非官方授权清单"]
    # file_dirs = ["./corpus/文本材料/对比"]
    # file_dirs = ["./corpus/文本材料-终版/官方授权清单", "./corpus/文本材料-终版/非官方授权清单"]
    file_dirs = ["./corpus/文本材料-终版/非官方授权清单"]
    for file_dir in file_dirs:
        fns = os.listdir(file_dir+"/en")
        # fns = ["German Industry and Global Enterprise BASF The History of a Company by Werner Abelshauser, Wolfgang von Hippel, Jeffrey Allan Johnson, Raymond G. Stokes (z-lib.org)_.txt"]
        # fns = ["Harry Potter and the Deathly Hallows (Book 7) by J. K. Rowling (z-lib.org)"]
        for fn in fns:
            if fn.find(".txt") < 0:
                continue
            book = []
            try:
                for line in open(file_dir+"/en/" + fn, "r", encoding="utf8"):
                    # print(1)
                    if line != "\n":
                        line = line.replace("\xad", "")
                        book.append(line)
            except:
                # i = 0
                for line in open(file_dir+"/en/" + fn, "r", encoding="gb18030"):
                    # print(2, i+1)
                    # i += 1
                    if line != "\n":
                        line = line.replace("\xad", "")
                        book.append(line)

            with open(file_dir+"/en/cut_text/" + fn[:-4] + "-cut.txt", "w", encoding="utf8") as f:
                for line in book:
                    seg_list = merge(replace_abbreviations(line).split())
                    f.write('\n'.join(seg_list)+"\n")

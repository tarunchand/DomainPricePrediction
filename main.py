import pandas as pd
import warnings
import re

from spellchecker import SpellChecker

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')


class TNode:
    def __init__(self):
        self.child = {}
        self.isWordFinished = False


class Trie:
    def __init__(self):
        self.root = TNode()

    def insert(self, word: str) -> None:
        word = word.upper()
        word = word[::-1]
        cur = self.root
        for i in word:
            if i not in cur.child:
                cur.child[i] = TNode()
            cur = cur.child[i]
        cur.isWordFinished = True

    def search(self, word: str) -> bool:
        word = word.upper()
        word = word[::-1]
        cur = self.root
        for i in word:
            if i not in cur.child:
                return False
            cur = cur.child[i]
        return cur.isWordFinished

    def getBrokenWords(self, word):
        word = word.upper()
        word_len = len(word)
        if word_len < 3:
            return [word]
        dp = [[False, -1]] * word_len
        for i in range(1, word_len):
            if word[i] not in self.root.child:
                continue
            cur = self.root.child[word[i]]
            for j in range(i - 1, -1, -1):
                if word[j] not in cur.child:
                    break
                cur = cur.child[word[j]]
                if cur.isWordFinished and j == 0:
                    dp[i] = [True, 0]
                    break
                if cur.isWordFinished and (j > 0) and dp[j - 1][0]:
                    dp[i] = [True, j]
        if not dp[word_len - 1][0]:
            return [word]
        broken_words = []
        i = word_len
        j = dp[i - 1][1]
        while i > 0:
            broken_words.append(word[j:i])
            i = j
            if j <= 0:
                break
            j = dp[i - 1][1]
        return broken_words[::-1]


def get_edit_distance(str1, str2):
    if str1 == str2:
        return 0

    len1 = len(str1)
    len2 = len(str2)

    dp = [[0 for i in range(len1 + 1)]
          for j in range(2)];

    for i in range(0, len1 + 1):
        dp[0][i] = i

    for i in range(1, len2 + 1):

        for j in range(0, len1 + 1):

            if (j == 0):
                dp[i % 2][j] = i

            elif str1[j - 1] == str2[i - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1]

            else:
                dp[i % 2][j] = (1 + min(dp[(i - 1) % 2][j],
                                        min(dp[i % 2][j - 1],
                                            dp[(i - 1) % 2][j - 1])))

    return dp[len2 % 2][len1]


def main1():
    domain_dataset = pd.read_csv('coursework_data.csv')
    domain_dataset['Length'] = [0] * 10000
    domain_dataset['CorrectedWord'] = [''] * 10000
    domain_dataset['EditDistance'] = [0] * 10000
    for i in range(10000):
        domain_dataset['Length'][i] = len(domain_dataset['Domain'][i].split('.')[0])
        domain_dataset['EditDistance'][i] = len(domain_dataset['Domain'][i].split('.')[0])
    spell = SpellChecker()
    for i in range(10000):
        domain_name = str(domain_dataset['Domain'][i].split('.')[0])
        if domain_name.isnumeric():
            continue
        corrected_word = spell.correction(domain_name)
        if corrected_word is not None:
            domain_dataset['CorrectedWord'][i] = corrected_word
            domain_dataset['EditDistance'][i] = get_edit_distance(domain_name, corrected_word)
        percentage = ((i + 1) * 100) / 10000
        print(str(percentage) + '%')
    domain_dataset.to_csv('result.csv', encoding='utf-8', index=False)


def main2():
    domain_dataset = pd.read_csv('coursework_data.csv')
    f = open('keywords.txt', 'w')
    for i in range(10000):
        domain_name = domain_dataset['Domain'][i].split('.')[0]
        f.write(domain_name + '\n')


def main3():
    domain_dataset = pd.read_csv('result_data.csv')
    keyword_dataset = pd.read_csv('Keyword_Search_Volume.csv', encoding="utf-16")
    keywords_length = len(keyword_dataset)
    search_volume = dict()
    for i in range(3, keywords_length):
        row = keyword_dataset['Keyword Stats 2022-11-29 at 14_10_47'][i].split('GBP')
        keywords = row[0].split()
        final_keyword = ''
        for word in keywords:
            final_keyword += word + '-'
        final_keyword = final_keyword[:-1]
        values = row[1].split()
        if len(values) > 0:
            search_volume[final_keyword] = float(values[0])
        else:
            search_volume[final_keyword] = 0
    domain_dataset['AvgMonthlySearchVolume'] = [0] * 10000
    search_volume['bullhead'] = 50000.0
    search_volume['p-i-x'] = 50000.0
    search_volume['v-l-c'] = 50000.0
    for i in range(10000):
        domain_name = re.sub('-+', '-', domain_dataset['Domain'][i].split('.')[0])
        try:
            domain_dataset['AvgMonthlySearchVolume'][i] = search_volume[domain_name]
        except KeyError as ex:
            print(ex)
    domain_dataset.to_csv('result.csv', encoding='utf-8', index=False)


def decision_tree(X, Y):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier()
    dtc.fit(X, Y)
    print(dtc.score(X, Y) * 100)


def pre_process():
    domain_dataset = pd.read_csv('final.csv')
    features = ['Length', 'EditDistance', 'AvgMonthlySearchVolume', 'ExtensionScore', 'isAlphabetsOnly']
    target_values = ['category']
    return domain_dataset[features], domain_dataset[target_values]


def svm_linear(X, Y):
    from sklearn import svm
    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)
    print(clf.score(X, Y))


def svm_poly(X, Y):
    from sklearn import svm
    clf = svm.SVC(kernel='poly', degree=8)
    clf.fit(X, Y)
    print(clf.score(X, Y))


def svm_gaussian(X, Y):
    from sklearn import svm
    clf = svm.SVC(kernel='rbf')
    clf.fit(X, Y)
    print(clf.score(X, Y))


def svm_sigmoid(X, Y):
    from sklearn import svm
    clf = svm.SVC(kernel='sigmoid')
    clf.fit(X, Y)
    print(clf.score(X, Y))


def split_string_by_special_characters(word):
    word = re.sub('[^a-zA-Z \n\.]', '-', word)
    res = [x for x in word.split('-') if len(x) > 0]
    return res


def get_corrected_words(word_list, spell):
    res = []
    for word in word_list:
        corrected_word = spell.correction(word)
        if corrected_word is None:
            corrected_word = ''
        res.append(corrected_word)
    return res


def process():
    domain_dataset = pd.read_csv('data/processed_dataset.csv')
    dictionary_words = open('final_dictionary.txt', 'r').read().split('\n')
    trie = Trie()
    for word in dictionary_words:
        trie.insert(word)
    spell = SpellChecker()
    for i in range(10005):
        res = []
        domain_name = str(domain_dataset['Domain'][i].split('.')[0])
        domain_name = split_string_by_special_characters(domain_name)
        if len(domain_name) == 1:
            broken_words = trie.getBrokenWords(domain_name[0])
            if len(broken_words) == 1:
                res = get_corrected_words(broken_words, spell)
            else:
                res = broken_words
        elif len(domain_name) > 1:
            res = get_corrected_words(domain_name, spell)
        domain_dataset['DictionaryWordsBroken'][i] = res
        print(i)
    domain_dataset.to_csv('processed_dataset.csv')


def temp():
    f = open('final_dictionary.txt').read().split('\n')
    count = 1
    w = open('new/' + str(count) + '.csv', 'w')
    w.write('Keyword' + '\n')
    for i in range(1, len(f)+1):
        w.write(f[i-1] + '\n')
        if i%7000 == 0:
            count += 1
            w = open('new/' + str(count) + '.csv', 'w')
            w.write('Keyword' + '\n')


if __name__ == '__main__':
    '''
    X, Y = pre_process()
    svm_linear(X, Y)
    '''
    temp()

from crawl import loadRPCDict, readLines, loadClsMap, distinctClsMap, write2txt, parallelMap, loadAddOnPatent, removeFile
from train import filterStop
from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import spacy
import numpy as np
import pandas as pd
import pytextrank
import yake
import pke
from string import punctuation
import matplotlib.pyplot as plt
from collections import Counter


nlp = spacy.load("en_core_web_lg")
# 添加pytextrank提取短语
nlp.add_pipe('textrank')



def parseAbstractAndDescription(html):
    soup = BeautifulSoup(html, 'html.parser')
    abst_soup = soup.select('section[itemprop=abstract]')
    desc_soup = soup.select('section[itemprop=description]')
    abstract = None if len(abst_soup) == 0 else abst_soup[0].get_text()
    descriptions = None if len(desc_soup) == 0 else desc_soup[0].get_text()
    
    return (abstract, descriptions)



def extractAbstAndDescParallel(patents):
    pool = ThreadPool()
    tuples = pool.map(extractAbstAndDesc, patents)
    pool.close()
    pool.join()
    # result = [vec_dict[doc] for doc in corpus]
    return tuples


def extractAbstAndDesc(patent):
    # print(patent)
    html = readLines("txt/" + patent + ".txt")
    (abst, desc) = parseAbstractAndDescription(html)
    if abst is None:
        print('patent:', patent, 'has none abst')
        abst = ''
    if desc is None:
        print('patent:', patent, 'has none desc')
        desc = ''
    if abst is None and desc is None:
        print("find null patent file:", patent)
        # removeFile('txt', patent, 'txt')

    return (patent, (abst, desc))


def filterRPC(rpc_dic, *prefixs):
    result = dict()
    for key, value in rpc_dic.items():
        for prefix in prefixs: 
            if value.startswith(prefix):
                result[key] = value
                break
            
            
    return result


# 所有的RPC分类号只分到第二个Level
def decreaseRPClevel(rpc_dic):
    result = dict()
    for key, value in rpc_dic.items():
        result[key] = value[0:3]
    return result

def load_csv_single_cloumn(file_name):
    df = pd.read_csv(file_name, header=None)
    rows = df.fillna('').values.tolist()

    words_list = []
    for row in rows:
        words = row[0]
        if len(words) == 0: continue
        
        text = words.split(',')
        for t in text:
            words_list.append(t.strip())
        
    return words_list
    
    
    
def load_key_phrase_csv():
    df = pd.read_csv('key_phrase.csv', header=None)
    rows = df.fillna('').values.tolist()
    phrase_dic = dict()
    for row in rows:
        patent = row[0]
        if len(patent) == 0: continue
        
        phrases = list()
        for i in range(1,4):
            text = row[i]
            # print(text)
            if len(text) == 0:
                continue
            strs = text.split(',')
            
            for s in strs:
                s = s.strip()
                if len(s) == 0: continue
                phrases.append(s.strip())
                
        if (len(phrases) == 0): continue
        phrase_dic[patent] = phrases
        
    return phrase_dic


def similarity(phrase_x, phrase_y):
    doc_x = nlp(phrase_x)
    doc_y = nlp(phrase_y)
    return doc_x.similarity(doc_y)


def wordCount(word, text):
    return (word, text.count(word))


def statisticWordFreqSpacyJob(rpc_class):

    
    rpc_dic = loadRPCDict()
    rpc_dic = filterRPC(rpc_dic, rpc_class)
    
    patents = list(rpc_dic.keys())
    
    
    print('Processing:', rpc_class)
    # 加载摘要和描述 (patent, (abst, desc))
    tuples = extractAbstAndDescParallel(patents)
    tuples = [t for t in tuples if t[1][0] is not None and t[1][1] is not None]
    print('Load text data done.')
    
    total = Counter()
    for (patent, (abst, desc)) in tuples:
        text = ' '.join([abst, desc])
        tokens = nlp(text.lower())
        
        tokens = filterStop(tokens)
        
        strs = [token.text for token in tokens]
        wordcount = Counter(strs)
        total.update(wordcount)
        print(wordcount.most_common(10))
        
    word_freq_dic = dict(total)
    np.save(rpc_class + '_word_freq_spacy.npy', word_freq_dic) 
    print(word_freq_dic)
    return (rpc_class, word_freq_dic)
    

def loadRPCWordFreqSpacy(rpc_class):
    file_name = rpc_class + '_word_freq_spacy.npy'
    
    word_freq_dic = np.load(file_name, allow_pickle=True)[()]
    return word_freq_dic
    
def statisticWordFreqSpacy():
    phrase_csv = load_key_phrase_csv() 
    
    rpc_class_list = list(phrase_csv.keys())
    
    # rpc_class_list = rpc_class_list[0:1]
    
    res = parallelMap(statisticWordFreqSpacyJob, rpc_class_list)
    
    dic = dict(res)
    
    # print(res)

def statisticWordCombination():
    
    rpc_dic = loadRPCDict()
    rpc_dic = filterRPC(rpc_dic, 'R01B01')
    patents = list(rpc_dic.keys())
    # patents = patents[0:10]
    tuples = extractAbstAndDescParallel(patents)
    
    comb = [['cache coherent', 'snoop'], ['cache coherent', 'directory']]
    count = [0, 0]
    for (patent, (abst, desc)) in tuples:
        text = ' '.join([abst, desc])
        for i in range(2):
            if comb[i][0] in text and comb[i][1] in text:
                count[i] += 1
    print('freeq of ', comb[0], 'is', count[0], '/', len(tuples))
    print('freeq of ', comb[1], 'is', count[1], '/', len(tuples))
    

def statisticWordFrequencyJob(tuple):
    (rpc_class, phrases) = tuple
    rpc_dic = loadRPCDict()
    rpc_dic = filterRPC(rpc_dic, rpc_class)
    
    patents = list(rpc_dic.keys())
    
    phrase_count = []
    
    print('Processing:', rpc_class)
    # 加载摘要和描述 (patent, (abst, desc))
    tuples = extractAbstAndDescParallel(patents)
    tuples = [t for t in tuples if t[1][0] is not None and t[1][1] is not None]
    print('Load text data done.')
    
    print('Start match phrase for', rpc_class)
    res_dic = { }
    for phrase in phrases:
        word_count = 0
        for (patent, (abst, desc)) in tuples:
            word_count += abst.count(phrase) + desc.count(phrase)
        phrase_count.append(word_count)
        print(phrase, word_count)
        res_dic[phrase] = word_count
        
    np.save(rpc_class + '_word_freq.npy', res_dic)
    
    return (rpc_class, res_dic)
    
def loadRPCWordFrequency(rpc_class):
    file_name = rpc_class + '_word_freq.npy'
    
    word_freq_dic = np.load(file_name, allow_pickle=True)[()]
    print(word_freq_dic)
    
    return word_freq_dic

# 统计单词在文本中出现的次数
def statisticWordFrequency():
    
    phrase_csv = load_key_phrase_csv()
    
    filter_set = set(['R01A03'])
    
    tuples = []
    for k, v in phrase_csv.items():
        if k in filter_set:
            tuples.append((k, v))
        
    
    
    # tuples = tuples[0:1]
        
    res_list = parallelMap(statisticWordFrequencyJob, tuples)

    print(res_list)
    return

def loadBookWords():
    word_list = []
    with open('book_gloss_undergrad.txt', encoding='utf-8') as file:
        for line in file.readlines():
            line = line.strip('\n').strip(',').strip()
            words = line.split(',')
            for word in words:
                word_list.append(word.strip())

    # with open('book_part_1.txt', encoding='utf-8') as file:
    #     for line in file.readlines():
    #         line = line.strip('\n').strip(',').strip()
    #         word_list.append(line)
        
    # with open('book_part_2.txt', encoding='utf-8') as file:
    #     word_list = []
    #     for line in file.readlines():
    #         line = line.strip('\n').strip(',').strip()
    #         word_list.append(line)

    return word_list


# 统计一个词汇表在一篇Patent里面出现的次数
def statisticWordFreqInPatent(tuple):
    (words_list, (patent, (abst, desc))) = tuple
    
    words_freq_dic = { }
    
    for word in words_list:
        text = ' '.join([abst, desc])
        cnt = text.count(word)
        if cnt == 0: continue
        
        if word in words_freq_dic.keys():
            words_freq_dic[word] += cnt
        else:
            words_freq_dic[word] = cnt
            
    print('.', end='', flush=True)
    return (patent, words_freq_dic)

    

def statisticWordFreqInAll(words_list):
    
    rpc_dic = loadRPCDict()
    # rpc_dic = filterRPC(rpc_dic, 'R01A01')
    
    patents = list(rpc_dic.keys())
    
    tuples = extractAbstAndDescParallel(patents)
    tmp  = []
    for tuple in tuples:
        tmp.append((words_list, tuple))
    tuples = tmp
    
    patent_word_freq_tuple_list = parallelMap(statisticWordFreqInPatent, tuples)
    
    
    rpc_classes = set(rpc_dic.values())
    rpc_word_freq_dic = { }
    for rpc_class in rpc_classes:
        rpc_word_freq_dic[rpc_class] = dict()
        

    patent_word_freq_dic = { }
    for (patent, words_freq_dic) in patent_word_freq_tuple_list:
        rpc_class = rpc_dic[patent]    
        rpc_word_freq_dic[rpc_class].update(words_freq_dic)
        patent_word_freq_dic[patent] = words_freq_dic
    
    
    print(rpc_word_freq_dic)
    
    
    np.save('arm_gloss_rpc_word_freq.npy', rpc_word_freq_dic)
    np.save('arm_gloss_patent_word_freq.npy', patent_word_freq_dic)
    # np.save('book_gloss_undergrad_rpc_word_freq.npy', rpc_word_freq_dic)
    # np.save('book_gloss_undergrad_patent_word_freq.npy', patent_word_freq_dic)
    

def drawWordFreqAllGraph():
    
    # file_name = 'arm_gloss_word_freq.npy'
    file_name = 'book_gloss_undergrad_word_freq.npy'
    
    word_freq_dic = np.load(file_name, allow_pickle=True)[()]
    print(word_freq_dic.keys())
    
    word_freq_dic.pop('R01F00')
    word_freq_dic.pop('R01G00')


    #根据RPC的字典序排序
    sorted_word_freq_dic = sorted(word_freq_dic.items(), key=lambda x : x[0], reverse=False)

    filter_words = ['read', 'write', 'modify', 'nit', 'AND', 'OR', 'NOT', 'NOR', 'bit']
    
    # 筛掉为词频为0的类别
    dataset = []
    for k, v in sorted_word_freq_dic:
        if len(v) == 0: continue
        v = filterWordFromDic(filter_words, v)
        dataset.append((k, v))
    
    start = 0
    while start < len(dataset):
        draw8GridGraph(dataset[start:start+8], 'Arm Datasheet & Book Glossary(Undergraduate), Top-25')
        start += 8
    return dataset
    
    
def filterWordFromDic(filter_list, dic:dict):
    for word in filter_list:
        if word in dic.keys():
            dic.pop(word)
    return dic


def printWordFreqToTxt(rpc_class, k):
    freq_dict = loadRPCWordFreqSpacy(rpc_class) 
    
    filter_words = ["fig", "data", "value", "result", "information", "example", "operation", "nodes"]
    for word in filter_words:
            freq_dict[word] = 0
            
    sorted_freq = sorted(freq_dict.items(), key=lambda x : x[1], reverse=True)

    top_k = sorted_freq[0:k]
    
    
    lines = []
    for word, count in top_k:
        line = word + ': ' + str(count) + '\n'
        lines.append(line)
        
    f = open(rpc_class + '_top_' + str(k) + '.txt', 'w')
    f.writelines(lines)
    f.close()
    

def loadRPCNameDict():
    df = pd.read_csv('rpc_name.csv', header=None)
    rows = df.fillna('').values.tolist()
    rpc_name_dic = dict()
    for row in rows:
        rpc = row[0]
        if len(rpc) == 0: continue
    
        rpc_name_dic[rpc] = row[1]
        
    return rpc_name_dic

    
def topKwords(word_count, k):
    
    sorted_count = sorted(word_count.items(), key=lambda x : x[1], reverse=True)
    
    top_k = sorted_count[0:k]

    x = []
    y = []
    for word, count in top_k:
        x.append(word)
        y.append(count)
    
    return x, y


rpc_name_dic = loadRPCNameDict()

# 根据传入的数据画图
def draw8GridGraph(dataset, title):
    n = 2
    m = 4
    
    fig, axes = plt.subplots(n, m)
    
    print('len: ', len(dataset))
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            
            if idx >= len(dataset): break

            (rpc_class, word_count) = dataset[idx]
            
            
            x, y = topKwords(word_count, 25)
            
            axes[i, j].bar(x, y)
            name = ''
            if rpc_class in rpc_name_dic.keys():
                name = '(' + rpc_name_dic[rpc_class] + ')'
            axes[i, j].set_title(rpc_class + name)
            # Rotate the x-axis label
            axes[i, j].tick_params(axis="x", labelrotation=90)

    plt.suptitle(title)
    plt.show()


def drawGraph():

    phrase_csv = load_key_phrase_csv() 

    rpc_class_set = set(phrase_csv.keys())
    
    except_set = set(['R01A14', 'R01A15', 'R01C03'])
    rpc_class_set -= except_set
    
    rpc_class_list = list(rpc_class_set)
    graph_data = {}
    
    k = 10
    
    filter_words = ["fig", "data", "value", "result", "information", "example", "operation", "nodes"]
    
    data_list = []
    for i in range(len(rpc_class_list)):
        rpc_class = rpc_class_list[i]
        
        freq_dict = loadRPCWordFrequency(rpc_class)
    
    
        if len(freq_dict) == 0:
            print('Skip empty dataset', rpc_class)
            continue
        
            
        sorted_freq = sorted(freq_dict.items(), key=lambda x : x[1], reverse=True)

        top_k = sorted_freq[0:k]
        

        x = []
        y = []
        for word, count in top_k:
            x.append(word)
            y.append(count)
        
        data_list.append((rpc_class, x, y))


    data_list = sorted(data_list, key=lambda x : x[0])

    
    n = 2
    m = 4
    
    index = 1
    
    fig, axes = plt.subplots(n, m)
    
    
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            
            (rpc_class, x, y) = data_list[idx]
            
            axes[i, j].bar(x, y)
            
            
            axes[i, j].set_title(rpc_class)

            # Rotate the x-axis label
            axes[i, j].tick_params(axis="x", labelrotation=45)
    

    plt.suptitle("keywords by hand")
    plt.show()
    
    
    fig, axes = plt.subplots(n, m)
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            
            (rpc_class, x, y) = data_list[idx + 8]
            
            axes[i, j].bar(x, y)
            
            
            axes[i, j].set_title(rpc_class)

            # Rotate the x-axis label
            axes[i, j].tick_params(axis="x", labelrotation=45)
           
    plt.suptitle("keywords by hand")
    plt.show()
    
    
def drawGraphSpacy():

    phrase_csv = load_key_phrase_csv() 

    rpc_class_set = set(phrase_csv.keys())
    
    except_set = set(['R01A14', 'R01A15', 'R01C03'])
    rpc_class_set -= except_set
    
    rpc_class_list = list(rpc_class_set)
    graph_data = {}
    
    k = 10
    
    filter_words = ["fig", "data", "value", "result", "information", "example", "operation", "nodes"]
    
    
    
    data_list = []
    for i in range(len(rpc_class_list)):
        rpc_class = rpc_class_list[i]
        
        freq_dict = loadRPCWordFreqSpacy(rpc_class)
    
    
        if len(freq_dict) == 0:
            print('Skip empty dataset', rpc_class)
            continue
        
        for word in filter_words:
            freq_dict[word] = 0
            
        sorted_freq = sorted(freq_dict.items(), key=lambda x : x[1], reverse=True)

        top_k = sorted_freq[0:k]
        

        x = []
        y = []
        for word, count in top_k:
            x.append(word)
            y.append(count)
        
        data_list.append((rpc_class, x, y))


    data_list = sorted(data_list, key=lambda x : x[0])

    
    n = 4
    m = 4
    
    index = 1
    
    fig, axes = plt.subplots(n, m)
    
    
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            
            (rpc_class, x, y) = data_list[idx]
            
            axes[i, j].bar(x, y)
            
            
            axes[i, j].set_title(rpc_class)

            # Rotate the x-axis label
            axes[i, j].tick_params(axis="x", labelrotation=45)

            
    # for rpc_class in rpc_class_list:
    #     if index > n * m: break

    #     x, y = graph_data[rpc_class]
    #     fig, ax = plt.subplots()
    #     plt.subplot(4, 4, index)
    #     plt.bar(x, y)
    #     fig.autofmt_xdate(rotation=45)
    #     plt.title(rpc_class)

    #     index += 1


    plt.suptitle("Word Count For RPC")
    plt.show()
    
    

def statisticOverlap():
    phrase_dic_csv = load_key_phrase_csv()
    r01a01 = phrase_dic_csv['R01A01']
    # print(r01a01)
    phrase_dic_patent = load_key_phrase('key_phrase_dict_100_R01A01.npy')
    
    rpc_dic = loadRPCDict()
    rpc_keywords = { }
    for patent, keywords in phrase_dic_patent.items():
        rpc = rpc_dic[patent]
        if rpc not in rpc_keywords.keys():
            rpc_keywords[rpc] = set()
        for kw in keywords:
            rpc_keywords[rpc].add(kw[0]) 
    # print(rpc_keywords)
    print('Num of rpc_dic.values():', len(rpc_keywords['R01A01']))
    
    sim_phrase_dic = { }
    i = 1
    for phrase_csv in r01a01:
        if i > 2: break
        j = 1
        for phrase_patent in rpc_keywords['R01A01']:
            print(i, "/", len(r01a01), ',',  j, '/', len(rpc_keywords['R01A01']))
            sim = similarity(phrase_csv, phrase_patent)
            print(phrase_csv, phrase_patent, sim)
            if sim > 0.9:
                
                if phrase_csv in sim_phrase_dic.keys():
                    sim_phrase_dic[phrase_csv].append(phrase_patent)
                else:
                    sim_phrase_dic[phrase_csv] = [phrase_patent]
            j += 1
        i += 1
        
    print(sim_phrase_dic)


# load an npy file
def load_key_phrase(file_name):
    phrase_dic = np.load(file_name, allow_pickle=True)[()]
    print(len(phrase_dic))
    # ========统计RPC分类下的关键词=========
    # rpc_dic = loadRPCDict()
    # rpc_keywords = { }
    # for patent, keywords in phrase_dic.items():
    #     rpc = rpc_dic[patent]
    #     if rpc not in rpc_keywords.keys():
    #         rpc_keywords[rpc] = set()
    #     for kw in keywords:
    #         rpc_keywords[rpc].add(kw[0])
            
    # for key, value in rpc_keywords.items():
    #     print(key)
    #     print(list(value)[0:30])
    # =====================================
    
    return phrase_dic

if __name__ == "__main__":


    
    # statisticOverlap()
    # statisticWordFrequency()
    # loadRPCWordFrequency('R01A01')
    # statisticWordFreqSpacy()
    
    # drawGraphSpacy()
    # drawGraph()
    
    # statisticWordCombination()
    
    # print(len(loadBookWords()))
    # words_list_book = loadBookWords()
    # words_list_arm = load_csv_single_cloumn('arm_gloss.csv')
    # words_list = list(set(words_list_arm + words_list_book))
    # statisticWordFreqInAll(words_list)
    
    drawWordFreqAllGraph()

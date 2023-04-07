from crawl import loadRPCDict, readLines, loadClsMap, distinctClsMap, write2txt, parallelMap, loadAddOnPatent, removeFile
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



# initialize keyphrase extraction model, here TopicRank
extractor = pke.unsupervised.TopicRank()


language = "en"
max_ngram_size = 3
deduplication_threshold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 20

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
# kw_extractor = yake.KeywordExtractor()


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
    

def get_mean_vec(li):
    res = []
    for i in range(len(li[0])):
        sum = 0
        for j in range(len(li)):
            sum += li[j][i]
        res.append(sum / len(li))
        
    return res


# tupe = (patent, (abst, desc))
def mapPatentText2Keywords(tuple):
    keywords = phrase_extract_yake(tuple[1][0] + ' ' + tuple[1][1])
    print('.', end='', flush=True)
    return (tuple[0], keywords)


def buildKeyPhraseFeature():
    phrase_set = load_key_phrase_csv()
    kw_dic = load_key_phrase('key_phrase_dict.npy')
    kw_tuples = []
    for key, value in kw_dic.items():
        kw_tuples.append((key, value, phrase_set))


    cls_map = distinctClsMap(loadClsMap())

    # 将所有的CPC分类号转换为长度为4的特征向量
    for key, value in cls_map.items():
        # 注意，可能会存在没有CPC分类号的情况
        if len(value) == 0:
            print(key)
            continue
        new_value = []
        for cpc in value:
            new_value.append(encodeCPC(cpc))
        # 将cpc的vectors取均值
        feat = []
        for i in range(4):
            sum = 0
            for j in range(len(new_value)):
                sum += new_value[j][i]
            avg = sum / len(new_value)
            feat.append(avg)
        cls_map[key] = feat
    
    # word2vec嵌入特征
    print("ready to word2vec...")
    vecs = keywords2vec(kw_tuples)
    feat_dic = dict(vecs)
    
        # 数据集里面加入cpc分类号
    for pantent, feature in feat_dic.items():
        cls = cls_map[pantent]
        feat_dic[pantent] = (cls, feature)
        
    np.save('key_phrase_feature_dict_5weight.npy', feat_dic)
    
def buildKeyPhraseData():
    # rpc_dic = loadRPCDict()
    rpc_dic = loadAddOnPatent()
    
    dataset_file = 'key_phrase_dict.npy'
    dataset = load_key_phrase(dataset_file)

    # 过滤要处理的rpc
    # rpc_dic = filterRPC(rpc_dic, 'R01A09')
    
    
    
    keys = list(set(rpc_dic.keys()) - set(dataset.keys()))
    print('Num of patent to process', len(keys))
    
    # 加载摘要和描述 (patent, (abst, desc))
    tuples = extractAbstAndDescParallel(keys)
    tuples = [t for t in tuples if t[1][0] is not None and t[1][1] is not None]
    
    rpc_keywords = { }
    
    # (patent, kw_list)
    kw_tuples = parallelMap(mapPatentText2Keywords, tuples)

    kw_dict = dict(kw_tuples)
    
    
    dataset.update(kw_dict)

    np.save(dataset_file, dataset)
    
    # ========统计RPC分类下的关键词=========
    for patent, keywords in kw_tuples:
        rpc = rpc_dic[patent]
        if rpc not in rpc_keywords.keys():
            rpc_keywords[rpc] = set()
        for kw in keywords:
            rpc_keywords[rpc].add(kw[0])
            
    for key, value in rpc_keywords.items():
        print(key)
        print(value)
        print('\n')
    # =====================================
    
    
    

# 构建npy数据集
# rpc_dic.npy 存储专利号到RPC号的映射
def buildDataset():
    
    # rpc_dic = loadRPCDict()
    rpc_dic = loadAddOnPatent()
    
    cls_map = distinctClsMap(loadClsMap())

    # 将所有的CPC分类号转换为长度为4的特征向量
    for key, value in cls_map.items():
        # 注意，可能会存在没有CPC分类号的情况
        if len(value) == 0:
            print(key)
            continue
        new_value = []
        for cpc in value:
            new_value.append(encodeCPC(cpc))
        # 将cpc的vectors取均值
        feat = []
        for i in range(4):
            sum = 0
            for j in range(len(new_value)):
                sum += new_value[j][i]
            avg = sum / len(new_value)
            feat.append(avg)
        cls_map[key] = feat

    # 过滤一下要处理的分类
    # rpc_dic = filterRPC(rpc_dic, 'R01')
    # print('Size of R01 & R02:', len(rpc_dic))
    
    # 开始处理摘要和描述
    keys = list(rpc_dic.keys())
    keys = keys[0:100]
    
    # 加载摘要和描述 (patent, (abst, desc))
    tuples = extractAbstAndDescParallel(keys)
    tuples = [t for t in tuples if t[1][0] is not None and t[1][1] is not None]

    # word2vec嵌入特征
    print("ready to word2vec...")
    vecs = word2vec(tuples)
    feat_dic = dict(vecs)
    
    
    # print(feat_dic)

    
    # 数据集里面加入cpc分类号

    
    for pantent, (abst_vec, desc_vec) in feat_dic.items():
        cls = cls_map[pantent]
        feat_dic[pantent] = (cls, abst_vec, desc_vec)
    np.save('feature_dict.npy', feat_dic)
    np.save('rpc_dict.npy', rpc_dic)
    
    return
    

# 使用pke提取关键词组
def phrase_extract_pke(text):
    # load the content of the document, here document is expected to be a simple 
    # test string and preprocessing is carried out using spacy
    extractor.load_document(input=text, language='en')

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    extractor.candidate_selection()

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 10 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=10)
    
    print(keyphrases)


def phrase_extract_yake(text):
    keywords = custom_kw_extractor.extract_keywords(text)
    print('.', end='', flush=True)
    return keywords
    

# 将CPC分类号进行编码
# 如 G06F2221 -> G 06 F 2221 -> [6, 6, 5, 2221]
def encodeCPC(cpc):
    a = cpc[0:1]
    b = cpc[1:3]
    c = cpc[3:4]
    d = cpc[4: ]
    return [ord(a) - ord('A'), int(b), ord(c) - ord('A'), int(d)]
    
    
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
        
    # examine the top-ranked phrases in the document
    # print('patent num: ', patent, 'Abstract')
    # tokens = nlp(abst.lower())
    # tuples = list()
    # for phrase in tokens._.phrases[:30]:
    #     print("phrase: {:<64}  score: {:<32}, count: {}".format(phrase.text, phrase.rank, phrase.count))
    #     tuples.append((phrase.text, phrase.count))
            
    # print('patent num: ', patent, 'Description')
    # tokens = nlp(desc.lower())

    # for phrase in tokens._.phrases[:30]:
    #     print("phrase: {:<64}  score: {:<32}, count: {}".format(phrase.text, phrase.rank, phrase.count))
    #     tuples.append((phrase.text, phrase.count))
        
    return (patent, (abst, desc))

# (patent, kw_list)
def get_kw_vector(tuple):
    patent = tuple[0]
    kw_list = tuple[1]
    phrase_set = tuple[2]

    # 计算权重之和 yake评分越低 相关性越高 所以用 1 - x
    total_weight = 0.
    for kw in kw_list:
        total_weight += (1 - kw[1])
        
    # 向量乘以权重
    res_vec = None
    for kw in kw_list:
        text = kw[0]
        weight = (1 - kw[1]) / total_weight
        # 增加权重占比
        if kw in phrase_set:
            weight *= 5
        doc = nlp(text)
        vector = doc.vector * weight
        if res_vec is None:
            res_vec = vector
        else:
            res_vec += vector
    
    print('.', end='', flush=True)
    return (patent, res_vec)
    

# (patent, kw_list)
def keywords2vec(tuple_list):
    vectors = parallelMap(get_kw_vector, tuple_list)
    return vectors
    
    
def word2vec(tuple_list):
    pool = ThreadPool()
    vectors = pool.map(get_mean_vect, tuple_list)
    pool.close()
    pool.join()
    # vector: (patent, (abst_vec, desc_vec))
    return vectors

# 将一个7位二进制向量转为int类型
def bin2int(bin_vec):
    res = 0
    for i in range(len(bin_vec)):
        res += bin_vec[i] << i
    return res

def lowbit(x):
    return x & -x

# 将一个0-127的整数转换为一个7位二进制向量
def int2bin(int_val, bound):
    while bound - lowbit(bound) != 0:
        bound -= lowbit(bound)
    
    bin_vec = []
    while bound != 0:
        bin_vec.append(int_val & 1)
        int_val = int_val >> 1
        bound = bound >> 1
    return bin_vec

# 从token序列中去除停用词和标点等
def filterStop(tokens):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB'] # 1
    for token in tokens:
        # 3 去除停用词和标点
        if token.is_stop or token.is_punct or token.like_num or not token.has_vector:
            continue
        # 4 专有名词、形容词、名词、动词
        if(token.pos_ in pos_tag):
            result.append(token)
    return result # 5

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# 摘要和描述的获取平均向量
def get_mean_vect(patent_tuple):
    patent = patent_tuple[0]
    (abst, desc) = patent_tuple[1]

    tokens = nlp(abst.lower())

    # for chunk in tokens.noun_chunks:
    #     print(chunk.text)
        
    tokens = filterStop(tokens)
    print(tokens)
    
    # abst_mean = tokens.vector
    # 归一化平均向量
    abst_vectors = [token.vector for token in tokens]
    abst_tmp = normalization(np.array(abst_vectors))
    abst_tmp = np.mean(abst_tmp, 0)
    
    tokens = nlp(desc.lower())
    tokens = filterStop(tokens)
    # desc_mean = tokens.vector
    desc_vectors  = [token.vector for token in tokens]
    # 归一化平均向量
    desc_tmp = normalization(np.array(desc_vectors))
    desc_tmp = np.mean(desc_tmp, 0)

    # tmp = np.vstack((abst_tmp, desc_tmp))
    print('.', end='', flush=True)
    return (patent, (abst_tmp, desc_tmp))

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
        

    for (patent, words_freq_dic) in patent_word_freq_tuple_list:
        rpc_class = rpc_dic[patent]    
        rpc_word_freq_dic[rpc_class].update(words_freq_dic)
    
    
    print(rpc_word_freq_dic)
    
    
    np.save('book_gloss_undergrad_word_freq.npy', rpc_word_freq_dic)
    

def drawWordFreqAllGraph():
    
    # file_name = 'arm_gloss_word_freq.npy'
    file_name = 'book_gloss_undergrad_word_freq.npy'
    
    word_freq_dic = np.load(file_name, allow_pickle=True)[()]
    print(word_freq_dic.keys())
    
    word_freq_dic.pop('R01F00')
    word_freq_dic.pop('R01G00')


    #根据RPC的字典序排序
    sorted_word_freq_dic = sorted(word_freq_dic.items(), key=lambda x : x[0], reverse=False)

    filter_words = ['read', 'write', 'modify', 'nit', 'AND', 'OR', 'NOT', 'NOR']
    
    # 筛掉为词频为0的类别
    dataset = []
    for k, v in sorted_word_freq_dic:
        if len(v) == 0: continue
        v = filterWordFromDic(filter_words, v)
        dataset.append((k, v))
    
    start = 0
    while start < len(dataset):
        draw8GridGraph(dataset[start:start+8], 'Book Glossary(Undergraduate), Top-20')
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
    
def printWordFreq():
    top_k = 100
    rpc_class = ['R01A01', 'R01A02', 'R01A03', 'R01A04']
    for rpc in rpc_class:
        printWordFreqToTxt(rpc, top_k)
    
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
            
            
            x, y = topKwords(word_count, 20)
            
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

# 加载数据集
def load_kw_dataset():
    # syntactic sugar [()] convert ndarry to dictionary
    # x[a, b, c] is syntactic sugar for x[(a, b, c)], so x[1,2] means getting the element at index (1,2) of the 2d array x. Because x is a 0d array, so x[()] is get the item of the 0-d arary x.
    feat_dic = np.load('key_phrase_feature_dict.npy', allow_pickle=True)[()]
    rpc_dic = np.load('rpc_dict.npy', allow_pickle=True)[()]
    
    # 加入额外的数据集
    # add_dic = loadAddOnPatent()
    # print('primitive data size:', len(rpc_dic), 'size of feat dic:', len(feat_dic))
    # print('add on data size:', len(add_dic))
    # rpc_dic.update(add_dic)
    
    
    # 减少分类层次或者过滤特定分类
    rpc_dic = decreaseRPClevel(rpc_dic)
    # rpc_dic = filterRPC(rpc_dic, 'R01A08', 'R01A09', 'R01A10', 'R01A11', 'R01A12', 'R01A13', 'R01A14', 'R01A15')
    # rpc_dic = filterRPC(rpc_dic, 'R01')
    print('filtered data size:', len(rpc_dic))
    
    
    rpc_list = list(set(rpc_dic.values()))
    rpc_list.sort()
    
    print(len(rpc_list), rpc_list)
    
    rpc_index = dict()
    X = []
    Y = []
    for i in range(len(rpc_list)):
        rpc_index[rpc_list[i]] = i
    
    ignored = 0
    zero_cpc = 0
    for patent, (cpc_vec, feature_vec) in feat_dic.items():
        if len(cpc_vec) == 0 or patent not in rpc_dic.keys():
            if (cpc_vec == 0):
                zero_cpc += 1
            ignored += 1
            continue
        
        rpc = rpc_dic[patent]
        
        # 将rpc转为独热编码或者二进制编码
        rpc_vec = int2bin(rpc_index[rpc], len(rpc_list))
        # print(len(rpc_vec))
        # rpc_vec = [0] * len(rpc_list)
        # rpc_vec[rpc_index[rpc]] = 1
        
        
        # X.append(feat.flatten())
        # feat 是个三元组(cpc_vec, abst_vec, desc_vec)
        # 连接成(604, )
        
        feat = np.concatenate((cpc_vec, feature_vec), axis=0)
        # feat = np.concatenate((cpc_vec, abst_vec, desc_vec), axis=0)
        # feat = doc_vec
        
        # if len(X) > 0 and X[0].shape != feat.shape:  
        #     print(feat.shape)
        #     continue
        X.append(feat)
        Y.append(rpc_vec)
    print('ignored data size:', ignored, 'zero cpc:', zero_cpc)
    print(X[0].shape)
    return X, Y

# 加载数据集
def load_dataset():
    # syntactic sugar [()] convert ndarry to dictionary
    # x[a, b, c] is syntactic sugar for x[(a, b, c)], so x[1,2] means getting the element at index (1,2) of the 2d array x. Because x is a 0d array, so x[()] is get the item of the 0-d arary x.
    feat_dic = np.load('feature_dict.npy', allow_pickle=True)[()]
    rpc_dic = np.load('rpc_dict.npy', allow_pickle=True)[()]
    
    # rpc_dic = decreaseRPClevel(rpc_dic)
    # rpc_dic = filterRPC(rpc_dic, 'R01', 'R02')
    
    rpc_list = list(set(rpc_dic.values()))
    rpc_list.sort()
    
    print(len(rpc_list), rpc_list)
    
    rpc_index = dict()
    X = []
    Y = []
    for i in range(len(rpc_list)):
        rpc_index[rpc_list[i]] = i
        
    for patent, (cpc_vec, abst_vec, desc_vec) in feat_dic.items():
        if len(cpc_vec) == 0 or patent not in rpc_dic.keys():
            continue
        
        rpc = rpc_dic[patent]
        
        # 将rpc转为独热编码或者二进制编码
        rpc_vec = int2bin(rpc_index[rpc], len(rpc_list))
        # print(len(rpc_vec))
        # rpc_vec = [0] * len(rpc_list)
        # rpc_vec[rpc_index[rpc]] = 1
        
        
        # X.append(feat.flatten())
        # feat 是个三元组(cpc_vec, abst_vec, desc_vec)
        # 连接成(604, )
        
        
        # cpc_vec = normalization(cpc_vec)
        doc_vec = (abst_vec + desc_vec) / 2
        feat = np.concatenate((cpc_vec, doc_vec), axis=0)
        # feat = np.concatenate((cpc_vec, abst_vec, desc_vec), axis=0)
        # feat = doc_vec
        
        # if len(X) > 0 and X[0].shape != feat.shape:  
        #     print(feat.shape)
        #     continue
        X.append(feat)
        Y.append(rpc_vec)
    print(X[0].shape)
    return X, Y

def train():
    X, Y = load_kw_dataset()
    # Split dataset to 8:2
    X_train, X_test, Y_train ,Y_test = train_test_split(X, Y, test_size=0.2)
    print(len(X_train), len(Y_train), len(X_test), len(Y_test))
    cls_1 = DecisionTreeClassifier()
    cls_2 = RandomForestClassifier()
    cls_3 = MLPClassifier(max_iter=1000)
    cls_4 = KNeighborsClassifier()
    cls_1.fit(X_train, Y_train)
    cls_2.fit(X_train, Y_train)
    cls_3.fit(X_train, Y_train)
    cls_4.fit(X_train, Y_train)
    Y_1_pred = cls_1.predict(X_test)
    Y_2_pred = cls_2.predict(X_test)
    Y_3_pred = cls_3.predict(X_test)
    Y_4_pred = cls_4.predict(X_test)
    # print(Y_pred)
    # print(Y_test)
    print("决策树测试集准确率:", accuracy_score(Y_test, Y_1_pred))
    print("随机森林测试集准确率:", accuracy_score(Y_test, Y_2_pred))
    print("多层感知机测试集准确率:", accuracy_score(Y_test, Y_3_pred))
    print("K近邻测试集准确率:", accuracy_score(Y_test, Y_4_pred))
          # "测试集精确率:", precision_score(Y_test, Y_pred),
          # "测试集召回率:", recall_score(Y_test, Y_pred)
    # print(f1_score(Y_test, Y_pred, average="macro"))
    # 0.666
    # print(f1_score(Y_test, Y_pred, average="micro"))
    # 0.8
    # print(f1_score(Y_test, Y_pred, average="weighted"))
    # 1.0
    # print(f1_score(Y_test, Y_pred, average="samples"))
    # 0.4

if __name__ == "__main__":
    # buildDataset()
    # buildKeyPhraseData()
    # buildKeyPhraseFeature()
    # print(load_key_phrase_csv())
    # print("数据集构建完成")

    # sorted_dic = sorted(word_dic.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_dic)
    
    # load_dataset()
    # load_kw_dataset()
    # load_key_phrase('key_phrase_dict.npy')

    
    # statisticOverlap()
    # statisticWordFrequency()
    # loadRPCWordFrequency('R01A01')
    # loadRPCWordFreqSpacy('R01A01')
    # statisticWordFreqSpacy()
    
    # drawGraphSpacy()
    # drawGraph()
    
    # printWordFreq()

    # statisticWordCombination()
    
    # print(len(loadBookWords()))
    # words_list = loadBookWords()
    # words_list = load_csv_single_cloumn('arm_gloss.csv')
    # statisticWordFreqInAll(words_list)
    
    drawWordFreqAllGraph()
    
    if bin2int(int2bin(89, 90)) != 89:
        print('test failed')
    # train()

import os
import requests
import re
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool as ThreadPool
from scipy.sparse import csr_matrix
from matplotlib.colors import LightSource
from matplotlib import cm
import seaborn as sns


file = "./download_.csv"
# headers = {
#     "User-Agent":'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
#     "cookie": 'SID=UwgL14SRVDopBAIWZbWjuT2X0M2Xv5AkJ92jtrAZgVZHZWoCQGasELrwFSyQCAgJkFeDEQ.; __Secure-1PSID=UwgL14SRVDopBAIWZbWjuT2X0M2Xv5AkJ92jtrAZgVZHZWoCXhqVo1cW_oRy2e1ec5HkLQ.; __Secure-3PSID=UwgL14SRVDopBAIWZbWjuT2X0M2Xv5AkJ92jtrAZgVZHZWoClesjnozJEzLWPlIGTMGQQA.; HSID=A5ZyVH--ZOEvWYKXS; SSID=AcWeaG4Fp7RL8PBXj; APISID=eDSX2RxHPHmyjeY_/AIxOdk7sJeZOgMhrW; SAPISID=5E8S0nlKexXu2Rdm/A2VX74ItukiG6dUDQ; __Secure-1PAPISID=5E8S0nlKexXu2Rdm/A2VX74ItukiG6dUDQ; __Secure-3PAPISID=5E8S0nlKexXu2Rdm/A2VX74ItukiG6dUDQ; SEARCH_SAMESITE=CgQI85cB; AEC=AUEFqZeqVY8bkdcK3aUeHC2AzCM3rE4zpT2SpSPdIvmNEKxffFuh_qrPOA; NID=511=DjFQmO_sQQOzFI4vGSCUlXPijFCcqN8OdVN5thtpYKx59CEp3WnvXuXcVAhOQ4x_0mCyfClzIs05N3FXZJRy_iev_3XNNLNMCq778dY3sABPpDnRS059PkrFoI-XDJ8p_xI9arUbSlX8_ZJPPa2WwB9Jfcmx6Sa08ZuRzbilJWK2TBHwwZFKxnDtffrWmVXiBgl9C62ZidD26kZi4kCB3guEtJZpSr4nmiXohItq182wa2kU1X3mc_AQX0AiYFkrYXPef3IB2MUQ7Zy-u9sA4O9vtg; 1P_JAR=2023-03-29-14; SIDCC=AFvIBn9QgQeU3ymEqjon-KqaE7wtZxuzUH17ejgms7tomjqEaGLBqZjZv5FATN5eiIKPBx-hZBI; __Secure-1PSIDCC=AFvIBn_rDVaa6gHVkp8L1Sjwol5A4lEog0cHayIEC6xJDZapkvZmzMzcFqzNpcV-wmqtmq-Me20; __Secure-3PSIDCC=AFvIBn8B92numYlijZf-C6-yU6uy-9KazzRN9KrE9ZtjGIej0HPt-eO4rbffyRHFhUZwC8PfvUY'
# }

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36', 
           "cookie": 'SID=UwgL14SRVDopBAIWZbWjuT2X0M2Xv5AkJ92jtrAZgVZHZWoCQGasELrwFSyQCAgJkFeDEQ.; __Secure-1PSID=UwgL14SRVDopBAIWZbWjuT2X0M2Xv5AkJ92jtrAZgVZHZWoCXhqVo1cW_oRy2e1ec5HkLQ.; __Secure-3PSID=UwgL14SRVDopBAIWZbWjuT2X0M2Xv5AkJ92jtrAZgVZHZWoClesjnozJEzLWPlIGTMGQQA.; HSID=A5ZyVH--ZOEvWYKXS; SSID=AcWeaG4Fp7RL8PBXj; APISID=eDSX2RxHPHmyjeY_/AIxOdk7sJeZOgMhrW; SAPISID=5E8S0nlKexXu2Rdm/A2VX74ItukiG6dUDQ; __Secure-1PAPISID=5E8S0nlKexXu2Rdm/A2VX74ItukiG6dUDQ; __Secure-3PAPISID=5E8S0nlKexXu2Rdm/A2VX74ItukiG6dUDQ; SEARCH_SAMESITE=CgQI85cB; AEC=AUEFqZeqVY8bkdcK3aUeHC2AzCM3rE4zpT2SpSPdIvmNEKxffFuh_qrPOA; NID=511=DjFQmO_sQQOzFI4vGSCUlXPijFCcqN8OdVN5thtpYKx59CEp3WnvXuXcVAhOQ4x_0mCyfClzIs05N3FXZJRy_iev_3XNNLNMCq778dY3sABPpDnRS059PkrFoI-XDJ8p_xI9arUbSlX8_ZJPPa2WwB9Jfcmx6Sa08ZuRzbilJWK2TBHwwZFKxnDtffrWmVXiBgl9C62ZidD26kZi4kCB3guEtJZpSr4nmiXohItq182wa2kU1X3mc_AQX0AiYFkrYXPef3IB2MUQ7Zy-u9sA4O9vtg; 1P_JAR=2023-03-29-14; SIDCC=AFvIBn9QgQeU3ymEqjon-KqaE7wtZxuzUH17ejgms7tomjqEaGLBqZjZv5FATN5eiIKPBx-hZBI; __Secure-1PSIDCC=AFvIBn_rDVaa6gHVkp8L1Sjwol5A4lEog0cHayIEC6xJDZapkvZmzMzcFqzNpcV-wmqtmq-Me20; __Secure-3PSIDCC=AFvIBn8B92numYlijZf-C6-yU6uy-9KazzRN9KrE9ZtjGIej0HPt-eO4rbffyRHFhUZwC8PfvUY',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
           'Accept-Language': 'en-US,en;q=0.5',
           'Accept-Encoding': 'zh-CN,zh;q=0.9,en;q=0.8',
           'referer': 'https://www.google.com/'
           }

def parallelMap(job, data):
    pool = ThreadPool()
    res = pool.map(job, data)
    pool.close()
    pool.join()
    return res

def getText(url):
    r = requests.get(url, headers=headers)
    # session = requests.Session()
    # r = session.get(url, timeout=30, headers=headers)  
    return r.text

def downloadPatentHtml(pub_num):
    url = 'https://patents.google.com/patent/' + pub_num + '/en\?oq\=' + pub_num
    text = getText(url)
    write2txt(text, "txt/" + pub_num + ".txt")
    print('download of ' + pub_num + ' end.')
    return text

def parseHtml(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.select('span[itemprop=title]')[0].string.strip()
    cls = soup.select('ul[itemprop=cpcs]')
    abstract = soup.select('div[class=abstract]')[0].string.strip()
    descriptions = soup.select('div[class=description]')
    claims = soup.select('div[class=claims]')
    citations = soup.select('tr[itemprop=backwardReferencesOrig]')
    #print(title)
    #print(cls)
    #print(abstract)
    #print(descriptions)
    #print(claims)
    print(citations)
    cites = parseCitations(soup)
    print(cites)

def parseCitations(patent):
    html = readLines("txt/" + patent + ".txt")
    soup = BeautifulSoup(html, 'html.parser')
    citations = soup.select('tr[itemprop=backwardReferencesOrig]')
    result = []
    for citation in citations:
        pub_num = citation.select("span[itemprop=publicationNumber]")[0].string.strip()
        result.append(pub_num)
    print("process " + patent + " done!")
    return (patent, result)

def parseClassifications(patent):
    html = readLines("txt/" + patent + ".txt")
    soup = BeautifulSoup(html, 'html.parser')
    ul = soup.select('ul[itemprop=cpcs]')
    res = []
    for u in ul:
        codes = u.select('span[itemprop=Code]')
        cls = []
        for code in codes:
            code_str = code.string.strip()
            cls.append(code_str)
        res.append(cls)
    print("process " + patent + " done!")
    return (patent, res)

# ClsMap.npy 存储的是专利号到CPC号的映射
def loadClsMap():
    cls_map = np.load('ClsMap.npy', allow_pickle=True).flat[0] # .flat[0] make sure it's dict
    return cls_map

# 对CPC分类号提取、去重
def distinctClsMap(cls_map):
    # 对CPC分类号提取、去重
    for key, value in cls_map.items():
        cpc_set = set()
        for cpcs in value:
            # 提取根据'/'拆分后的cpc分类号
            cpc_split = None
            for cpc in cpcs:
                if '/' in cpc:
                    cpc_split = cpc.split('/')[0]
                    break
            cpc_set.add(cpc_split)
        cls_map[key] = list(cpc_set)
    return cls_map

def buildClsMap():
    download_set = set(listFiles('txt', suffix='txt'))
    cur_cls_map = np.load('ClsMap.npy', allow_pickle=True).flat[0] # .flat[0] make sure it's dict
    # 增加未处理过的专利
    download_set = download_set - set(cur_cls_map.keys())
    # download_set = ["CA2809992A1", "US20040230751A1"]
    pool = ThreadPool()
    tuples = pool.map(parseClassifications, download_set)
    pool.close()
    pool.join()
    cls_dic = dict(tuples)
    for key, value in cls_dic.items():
        cur_cls_map[key] = value
        
    np.save('ClsMap.npy', cur_cls_map)

def loadCitationDic():
    cite_map = np.load('CiteMap.npy', allow_pickle=True).flat[0]
    return cite_map

def buildCitationDic():
    patent_list = pd.read_csv(file).values.tolist()
    patent_list = [patent[0] for patent in patent_list]
    tuples = parallelMap(parseCitations, patent_list)
    cite_dic = dict(tuples)
    print(cite_dic)
    np.save('CiteMap.npy', cite_dic)

def loadAddOnPatent():
    rpcs = listFiles('add_on_csv', 'csv')

    rpc_dic = { }
    for rpc in rpcs:
        rows = pd.read_csv('add_on_csv' + '/' + rpc + '.csv', header=None).values.tolist()
        for row in rows:
            # 去调Patent号中的横杠
            strs = row[0].split('-')
            # 修正可能存在的错误
            if strs[1].isdigit() and int(strs[1][0:4]) in range(1900, 2100):
                l = list(strs[1])
                l.insert(4, '0')
                strs[1] = ''.join(l)
            patent = ''.join(strs)
            # patent = row[0].replace('-', '')
            rpc_dic[patent] = rpc
        
    return rpc_dic

    
def loadRPCDict():
    arm_rpc_patents = pd.read_csv("arm_rpc.csv", header=None).values.tolist()
    mips_rpc_patents = pd.read_csv("mips_rpc.csv", header=None).values.tolist()
    rpc_patents = arm_rpc_patents + mips_rpc_patents
    rpc_dic = dict()
    for row in rpc_patents:
        rpc_dic[row[0]] = row[1]
    return rpc_dic

def buildClsDic():
    rpc_dic = loadRPCDict()
    cite_dic = loadCitationDic()
    cls_dic = loadClsMap()
    inner_set = set(rpc_dic.keys()) & set(cite_dic.keys())
    res = {}
    for patent in inner_set:
        rpc = rpc_dic[patent]
        cls = cls_dic[patent]
        if rpc not in res.keys():
            res[rpc] = []
        for c in cls:
            for s in c:
                if '/' in s and s.split('/')[1] == '00':
                    res[rpc].append(s)
    return res
    
def drawHeatGraph(cls_map):
    rpc_cls = list(cls_map.keys())
    cpc_cls = set()
    for v in cls_map.values():
        for cpc in v:
            cpc_cls.add(cpc)
    cpc_cls = list(cpc_cls)
    rpc_index = {}
    cpc_index = {}
    for i in range(len(cpc_cls)):
        cpc_index[cpc_cls[i]] = i
    for i in range(len(rpc_cls)):
        rpc_index[rpc_cls[i]] = i
    matrix = np.zeros([len(rpc_cls), len(cpc_cls)])
    for rpc in rpc_cls:
        cpcs = cls_map[rpc]
        for cpc in cpcs:
            matrix[rpc_index[rpc]][cpc_index[cpc]] += 1
    # matrix = csr_matrix(matrix).todense()
    fig, ax = plt.subplots()
    # im = ax.imshow(matrix)
    im = ax.spy(matrix, precision=0, markersize=2, aspect='auto', origin='lower')
    # ax.set_xticks(np.arange(len(rpc_cls)), labels=rpc_cls)
    # ax.set_yticks(np.arange(len(cpc_cls)), labels=cpc_cls)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.show()
    fig, ax = plt.subplots()
    x, y = matrix.nonzero()
    plt.scatter(y, x, s=10, c=matrix[x,y]) #color as the values in k matrix
    ax.set_xlabel('cpc')
    ax.set_ylabel('rpc')
    #ax.legend()
    plt.show()

def process():
    cls_map = buildClsMap()
    drawHeatGraph(cls_map)

def getHtml(rpc_patent_dic, down_dependency: bool):
    
    # 要下载的专利列表
    patent_list = rpc_patent_dic.keys()
    # patent_list = pd.read_csv(file).values.tolist()
    # patent_list = [row[0] for row in patent_list]
    patent_set = set(patent_list)
    
    # 已经下载过的专利文件
    download_set = set(listFiles('txt', 'txt'))
    
    # 下载列表是
    download_list = list(patent_set - download_set)
    
    parallelMap(downloadPatentHtml, download_list)
    # for patent in download_list:
    #     # 下载专利文本 更新已下载集合
    #     downloadPatentHtml(patent)
    #     download_set.add(patent)
    #     # 是否下载citation中的专利
    #     if down_dependency:
    #         getDependencyHtml(patent, download_set)

def getDependencyHtml(patent, download_set):
    (patent, citations) = parseCitations(patent)
    print(citations)
    for citation in citations:
        if citation in download_set:
            continue
        downloadPatentHtml(citation)
        download_set.add(citation)
        print('download citation ' + citation + ' end.')

def getPdf():
    patent_list = pd.read_csv(file).values.tolist()
    for row in patent_list:
        if row[1] == 1: continue
        patent = row[0]
        url = 'https://patents.google.com/patent/'+ patent + '/en?oq=' + patent
        text = getText(url)
        links = re.findall(r"https:\/\/patentimages.storage.googleapis.com\/.*\.pdf", text)
        print('downloading from: ' + links[0])
        f = open("data/" + patent + ".pdf", "wb")
        pdf = requests.get(links[0])
        f.write(pdf.content)
        f.close()

        print("download end.")
        row[1] = 1
        df = pd.DataFrame(patent_list)
        df.to_csv(file, index=False, header=False)

def listFiles(path, suffix):
    patents = os.listdir(path)
    num_list = [patent.split('.')[0] for patent in patents if patent.split('.')[1] == suffix]
    return num_list

def removeFile(path, name, suffix):
    os.remove(path + '/' + name + '.' + suffix)

def write2txt(text, file):
    with open(file, 'w') as f:
        f.write(text)

def readLines(file):
    with open(file, 'r') as f:
        text = f.read()
        return text

if __name__ == "__main__":
    url = 'https://patents.google.com/patent/US7107551B1/en'
    # write2txt(getText(url), 'html1.txt')
    # print(downloadPatentHtml('US20130292297A1'))
    getHtml(loadRPCDict(), False)
    # loadAddOnPatent()
    # buildClsMap()
    # loadClsMap()
    # buildCitationMap()
    # loadCitationMap()
    # loadRPC()
    # process()

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
headers = {
    "User-Agent":'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    "cookie": 'HSID=A-WgUv8-9biHpTC0Y; SSID=A363Mn1UclGReFBuW; APISID=ZnR0fyqMe9r_PSeb/A2-xnXn9Z2f3SsfKZ; SAPISID=1-I-MyZZBPuGLZxO/A0LQr6sWArPmzDsjh; __Secure-1PAPISID=1-I-MyZZBPuGLZxO/A0LQr6sWArPmzDsjh; __Secure-3PAPISID=1-I-MyZZBPuGLZxO/A0LQr6sWArPmzDsjh; SEARCH_SAMESITE=CgQIhZcB; SID=RggL17J222nsOCEGuLmBCDERGbXwrGN95eLFuQUG8HNfkgsO607nyWZwOgrh2ceO_WvxhQ.; __Secure-1PSID=RggL17J222nsOCEGuLmBCDERGbXwrGN95eLFuQUG8HNfkgsOVVpkjEs97E1DZ5gQNjVa7A.; __Secure-3PSID=RggL17J222nsOCEGuLmBCDERGbXwrGN95eLFuQUG8HNfkgsOnmAZMGSaIMpE9lAJeCVLcA.; AEC=AakniGMaPYXlf5w5JwXm2lAD89Hrszit6T7zdFfDcLsIuWjaX-MAymd3FA; NID=511=mSB05gSijon-elCht7GtUollUTlk4iD8Gm1OWOytbsODBBsjeHc8xFNKs3ayDWXO9dFUxL_FAAnoTjouvXYfUU3v2Hfp9YlI1xSAfWfcpsACBp5VLWerdYEl8awUKayzpqSlbuOuk2V3Zx7y8pXJrSs3CFCyW7HcA7I-EzmQH5YS5rJg9WLDMFPFDeJEuYRteQg1NKQk-Gv90Di297fmOHoKZZlc9Z24MhePMuCFYuO5QFo_5wiOIG76bFVyn6Vf_qTl7YVW1fa917TIbI-z1K5dr7CaojncYgwJvFdd_gB7w5mbktZIrMyFOPVRHaqFLDb_Vy-A3TcLXEJOySoH3A; 1P_JAR=2022-12-21-03; SIDCC=AIKkIs0bWVqaRzha2iDrB_N5YBkFVKZJasFOk2OE5ov4r6hPtCrL6IE2DC9jaB2FdVrD9T8Ke6A; __Secure-1PSIDCC=AIKkIs31mKR3APQC9OzT508_PCkALkg0ZwwNdNMuBx-2x9q50eFsJYlQ4bem3s7z3yTLj0AN4unT; __Secure-3PSIDCC=AIKkIs2OdZtYR9tWCjAfq6uykZygR8-CvQ6rzqxyUGP_YeG0LLB9h3Uea0wLs24usTdK6_BryKc'
}

def parallelMap(job, data):
    pool = ThreadPool()
    res = pool.map(job, data)
    pool.close()
    pool.join()
    return res

def getText(url):
    r = requests.get(url, headers=headers)
    # r = requests.session().get(url)
    return r.text

def downloadPatentHtml(pub_num):
    url = 'https://patents.google.com/patent/' + pub_num + '/en'
    text = getText(url)
    write2txt(text, "txt/" + pub_num + ".txt")
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

def loadClsDic():
    cls_map = np.load('ClsMap.npy', allow_pickle=True).flat[0] # .flat[0] make sure it's dict
    return cls_map

def buildClsDic():
    download_set = set(listPatents('txt'))
    # download_set = ["CA2809992A1", "US20040230751A1"]
    pool = ThreadPool()
    tuples = pool.map(parseClassifications, download_set)
    pool.close()
    pool.join()
    cls_dic = dict(tuples)
    print(cls_dic)
    np.save('ClsMap.npy', cls_dic)

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

def loadRPCDict():
    arm_rpc_patents = pd.read_csv("arm_rpc.csv").values.tolist()
    mips_rpc_patents = pd.read_csv("mips_rpc.csv").values.tolist()
    rpc_patents = arm_rpc_patents + mips_rpc_patents
    rpc_dic = dict()
    for row in rpc_patents:
        rpc_dic[row[0]] = row[1]
    return rpc_dic

def buildClsMap():
    rpc_dic = loadRPC()
    cite_dic = loadCitationDic()
    cls_dic = loadClsDic()
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

def getHtml():
    patent_list = loadRPCDict().keys()
    # patent_list = pd.read_csv(file).values.tolist()
    # patent_list = [row[0] for row in patent_list]
    patent_set = set(patent_list)
    download_set = set(listPatents('txt'))
    download_list = list(patent_set - download_set)
    for patent in download_list:
        if patent in download_set:
            html = readLines("txt/" + patent + ".txt")
        else:
            html = downloadPatentHtml(patent)
            download_set.add(patent)
            print('download of ' + patent + ' end.')
        getDependencyHtml(patent, download_set)

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

def listPatents(path):
    patents = os.listdir(path)
    num_list = [patent.split('.')[0] for patent in patents if patent.split('.')[1] == 'txt']
    return num_list

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
    getHtml()
    # buildClsMap()
    # loadClsMap()
    # buildCitationMap()
    # loadCitationMap()
    # loadRPC()
    # process()

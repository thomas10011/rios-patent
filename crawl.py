import os
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup

file = "./download.csv"
headers = {
    "User-Agent":'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    "cookie": 'HSID=A-WgUv8-9biHpTC0Y; SSID=A363Mn1UclGReFBuW; APISID=ZnR0fyqMe9r_PSeb/A2-xnXn9Z2f3SsfKZ; SAPISID=1-I-MyZZBPuGLZxO/A0LQr6sWArPmzDsjh; __Secure-1PAPISID=1-I-MyZZBPuGLZxO/A0LQr6sWArPmzDsjh; __Secure-3PAPISID=1-I-MyZZBPuGLZxO/A0LQr6sWArPmzDsjh; SEARCH_SAMESITE=CgQIhZcB; SID=RggL17J222nsOCEGuLmBCDERGbXwrGN95eLFuQUG8HNfkgsO607nyWZwOgrh2ceO_WvxhQ.; __Secure-1PSID=RggL17J222nsOCEGuLmBCDERGbXwrGN95eLFuQUG8HNfkgsOVVpkjEs97E1DZ5gQNjVa7A.; __Secure-3PSID=RggL17J222nsOCEGuLmBCDERGbXwrGN95eLFuQUG8HNfkgsOnmAZMGSaIMpE9lAJeCVLcA.; AEC=AakniGMaPYXlf5w5JwXm2lAD89Hrszit6T7zdFfDcLsIuWjaX-MAymd3FA; NID=511=mSB05gSijon-elCht7GtUollUTlk4iD8Gm1OWOytbsODBBsjeHc8xFNKs3ayDWXO9dFUxL_FAAnoTjouvXYfUU3v2Hfp9YlI1xSAfWfcpsACBp5VLWerdYEl8awUKayzpqSlbuOuk2V3Zx7y8pXJrSs3CFCyW7HcA7I-EzmQH5YS5rJg9WLDMFPFDeJEuYRteQg1NKQk-Gv90Di297fmOHoKZZlc9Z24MhePMuCFYuO5QFo_5wiOIG76bFVyn6Vf_qTl7YVW1fa917TIbI-z1K5dr7CaojncYgwJvFdd_gB7w5mbktZIrMyFOPVRHaqFLDb_Vy-A3TcLXEJOySoH3A; 1P_JAR=2022-12-21-03; SIDCC=AIKkIs0bWVqaRzha2iDrB_N5YBkFVKZJasFOk2OE5ov4r6hPtCrL6IE2DC9jaB2FdVrD9T8Ke6A; __Secure-1PSIDCC=AIKkIs31mKR3APQC9OzT508_PCkALkg0ZwwNdNMuBx-2x9q50eFsJYlQ4bem3s7z3yTLj0AN4unT; __Secure-3PSIDCC=AIKkIs2OdZtYR9tWCjAfq6uykZygR8-CvQ6rzqxyUGP_YeG0LLB9h3Uea0wLs24usTdK6_BryKc'
}


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

def parseCitations(soup):
    citations = soup.select('tr[itemprop=backwardReferencesOrig]')
    result = []
    for citation in citations:
        pub_num = citation.select("span[itemprop=publicationNumber]")[0].string.strip()
        result.append(pub_num)
    return result

def parseClassifications(soup):
    cls = soup.select('ul[itemprop=cpcs]')
    return cls

def getHtml():
    patent_list = pd.read_csv(file).values.tolist()
    download_set = set(listPatents('txt'))
    for row in patent_list:
        patent = row[0]
        if patent in download_set:
            html = readLines("txt/" + patent + ".txt")
        else:
            html = downloadPatentHtml(patent)
            download_set.add(patent)
            print('download of ' + patent + ' end.')
        getDependencyHtml(html, download_set)

def getDependencyHtml(html, download_set):
    soup = BeautifulSoup(html, 'html.parser')
    citations = parseCitations(soup)
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

url = 'https://patents.google.com/patent/US7107551B1/en'
# write2txt(getText(url), 'html1.txt')
getHtml()

text = readLines('html1.txt')
text2 = readLines('html2.txt')
parseHtml(text)
parseHtml(text2)

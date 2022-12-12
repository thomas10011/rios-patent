import requests
import re
import pandas as pd

file = "./download.csv"
patent_list = pd.read_csv(file).values.tolist()

for row in patent_list:
    if row[1] == 1: continue
    patent = row[0]
    url = 'https://patents.google.com/patent/'+ patent + '/en?oq=' + patent
    headers = {"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0'}
    r = requests.get(url, headers=headers)
    text = r.text
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



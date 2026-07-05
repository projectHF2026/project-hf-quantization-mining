import requests
import bs4
import time

file_URL = open('file--url.txt', 'w')

headers = {"authorization": f"TOKEN"}
#response = requests.request("GET", URL , headers=headers)

with open("dependent_trasformers.csv", 'w') as f:

    n_reqs = 0

    URL = "https://github.com/huggingface/transformers/network/dependents"
    #response = requests.get(URL)
    response = requests.request("GET", URL, headers=headers)
    n_reqs += 1
    time.sleep(60)

    file_URL.write(URL + "\n")

    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    print(response.status_code)

    a_text_bold = soup.findAll('a', {"class", "text-bold"})
    span = soup.findAll('span', {"class", "color-fg-muted text-bold pl-3"})

    i = 0
    for aa in a_text_bold:
        star = span[i].get_text().strip()
        forked = span[i+1].get_text().strip()
        f.write(aa.get("href") + "," + star + "," + forked + "\n")
        i += 1

    next_button = soup.findAll('a', {"class", "btn btn-outline BtnGroup-item"})

    #response = requests.get(next_button[0].get("href"))
    response = requests.request("GET", next_button[0].get("href"), headers=headers)
    n_reqs += 1
    time.sleep(60)

    file_URL.write(str(next_button[0].get("href")) + "\n")

    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    print(response.status_code)
    next_button = soup.findAll('a', {"class", "btn btn-outline BtnGroup-item"})

    while len(next_button) > 1:
        a_text_bold = soup.findAll('a', {"class", "text-bold"})
        span = soup.findAll('span', {"class", "color-fg-muted text-bold pl-3"})
        i = 0
        for aa in a_text_bold:
            star = span[i].get_text().strip()
            forked = span[i + 1].get_text().strip()
            f.write(aa.get("href") + "," + star + "," + forked + "\n")
            i += 1
        next_button = soup.findAll('a', {"class", "btn btn-outline BtnGroup-item"})

        #response = requests.get(next_button[1].get("href"))
        response = requests.request("GET", next_button[1].get("href"), headers=headers)
        n_reqs += 1
        time.sleep(60)

        file_URL.write(str(next_button[1].get("href")) + "\n")

        # if n_reqs % 100 == 0:
        #     time.sleep(120)
        #     print("stop waiting", n_reqs)
        # if n_reqs % 350 == 0:
        #     time.sleep(300)
        #     print("stop waiting every 499 for 5 minutes", n_reqs)

        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        print(response.status_code)
        
        if response.status_code > 200:
            print(n_reqs)
        next_button = soup.findAll('a', {"class", "btn btn-outline BtnGroup-item"})

    if len(next_button) == 1:
        #response = requests.get(next_button[0].get("href"))
        response = requests.request("GET", next_button[0].get("href"), headers=headers)

        file_URL.write(str(next_button[0].get("href")) + "\n")

        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        print(response.status_code)

        a_text_bold = soup.findAll('a', {"class", "text-bold"})
        span = soup.findAll('span', {"class", "color-fg-muted text-bold pl-3"})

        i = 0
        for aa in a_text_bold:
            star = span[i].get_text().strip()
            forked = span[i + 1].get_text().strip()
            f.write(aa.get("href") + "," + star + "," + forked + "\n")
            i += 1

file_URL.close()

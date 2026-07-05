# HF-models-scraper.py
# Script to scrape all models hosted on Hugging Face hub using HF API

import json
import requests
import time

num_reqs = 0

# request header for collecting models' data using HF API token
#headers = {"authorization": f"hf_YOUR_TOKEN_HERE"}
#headers = {"authorization": f"Bearer hf_YOUR_TOKEN_HERE"}
import os
HF_TOKEN = os.environ["HF_TOKEN"]
headers = {"authorization": f"Bearer {HF_TOKEN}"}

# retrieving JSON file containing all models hosted on HF hub
response = requests.request('GET', 'https://huggingface.co/api/models/', headers=headers)
num_reqs += 1

resp_headers_link = response.headers["Link"]
rel_value = resp_headers_link.split(";")[1].strip()
next_link = resp_headers_link.split(";")[0].replace("<", "").replace(">", "")

#print(rel_value)
#print(next_link)

modelLIST = json.loads(response.text)

# The base URL to get information for a specific model ( defined in {} )
API_URL = "https://huggingface.co/api/models/{}"

i = 0
while 'next' in rel_value:
    # reading the JSON file containg all HF models
    with open('/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/HuggingFaceStudy/modelsPerPages/page' + str(i) + ".json", 'w') as f:

        json.dump(modelLIST, f, indent=2)
        #modelsJSON = json.load(f)
        #print(modelsJSON)
        #print(len(modelsJSON))

        # for any HF model retrieve the corresponding ID (i.e., the model name)
        for data in modelLIST:
            modelID = data['id']
            print(modelID)

            # for any HF model download its information and store them into a file
            response = requests.request("GET", API_URL.format(modelID), headers=headers)
            num_reqs += 1

            if (num_reqs % 1000) == 0:
                time.sleep(1800)

            modelData = json.loads(response.text)
            with open('/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/HuggingFaceStudy/modelsInfo/' + modelID.replace("/", "£sep£") + ".json", 'w') as f_model:
                json.dump(modelData, f_model, indent=2)
            #print(json.dumps(data, indent=4))

        i += 1
        API_URL = next_link
        response = requests.request('GET', API_URL, headers=headers)
        num_reqs += 1

        if (num_reqs % 1000) == 0:
            time.sleep(1800)

        resp_headers_link = response.headers["Link"]
        rel_value = resp_headers_link.split(";")[1].strip()
        next_link = resp_headers_link.split(";")[0].replace("<", "").replace(">", "")
        modelLIST = json.loads(response.text)
        API_URL = "https://huggingface.co/api/models/{}"

        with open('/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/HuggingFaceStudy/modelsPerPages/page' + str(i) + ".json", 'w') as f:
            json.dump(modelLIST, f, indent=2)

# for any HF model retrieve the corresponding ID (i.e., the model name)
for data in modelLIST:
    modelID = data['id']
    print(modelID)

    # for any HF model download its information and store them into a file
    response = requests.request("GET", API_URL.format(modelID), headers=headers)
    num_reqs += 1

    if (num_reqs % 1000) == 0:
        time.sleep(1800)

    modelData = json.loads(response.text)
    with open('/scratch/oldhome/user/projects/JAW/scripts/icpc-approch/output_dir/HuggingFaceStudy/modelsInfo/' + modelID.replace("/", "£sep£") + ".json", 'w') as f_model:
        json.dump(modelData, f_model, indent=2)

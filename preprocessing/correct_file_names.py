#import numpy as np
import os
import regex as re
from shutil import copyfile
from pathlib import Path
import json

# folder="content/content_train/trainData/"
# Path(folder[:-1]).mkdir(parents=True, exist_ok=True)
# list_dir = sorted(os.listdir(folder))
# print(len(list_dir))
# for file in list_dir:
#     file_name = file[:-4]
#     file_name = re.sub(" ", "_", file_name)
#     file_name = re.sub("\.", "_", file_name)
#     file_name = re.sub("\-", "_", file_name)
#     os.rename(folder+file, folder+file_name+".jpg")

# json_file="content/content_train/trainJson.json"
# with open(json_file) as json_f:
#     json_load = json.load(json_f)
#     json_d = json.dumps(json_load)
#     file_names = re.findall(r'"file_name": "([^\,]*)"',json_d)
#     for name in file_names:
#         name = name[:-4]
#         new_name = re.sub(" ", "_", name)
#         new_name = re.sub("\.", "_", new_name)
#         new_name = re.sub("\-", "_", new_name)
#         new_name = new_name
#         new_json_d = re.sub(name, new_name, json_d)
#         json_d = new_json_d

# json_file2 = "content/content_train/trainJson2.json"
# with open(json_file2, "w") as file:
#     file.write(json_d)

problems=[]
json_file="content/content_train/trainJson.json"
with open(json_file) as json_f:
    json_load = json.load(json_f)
    json_d = json.dumps(json_load)
    file_names = re.findall(r'"file_name": "([^\,]*)"',json_d)
    for name in file_names:
        if os.path.exists("content/content_train/trainData/"+name):
            continue
        else:
            problems.append(name)
print(problems)


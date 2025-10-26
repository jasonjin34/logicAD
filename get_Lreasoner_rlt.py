"""
To output decision for each individual test sample.
"""

import os
from os import path as osp
import re

DATA_PATH='/Users/fengqihui/Documents/GitHub/logicAD/datasets/proofs'
# exps = ['breakfast_box', 'juice_bottle', 'splicing_connectors']
exps = ['juice_bottle']
TAG = '1508_v2'
proved_pattern = re.compile(r"\s*Exiting with (\d+) proofs?\.\s*")
unproved_pattern = re.compile(r"\s*Exiting with failure\.\s*")

for exp in exps:
    # print(exp)
    
    repo = osp.join(DATA_PATH, '{}_{}'.format(exp, TAG))
    file_list = []
    for root, dirs, files in os.walk(repo):
        for file in files:
            if '.out' in file:
                file_list.append(file)
    rlt = ""
    
    for file in file_list:
        decision_tag = -1
        with open(osp.join(repo, file), 'r') as f:
            for line in f:
                is_match = proved_pattern.search(line)
                if is_match:
                    decision_tag = 1
                    break
                is_match = unproved_pattern.search(line)
                if is_match:
                    decision_tag = 0
                    break
        rlt += "{} {}\n".format(file.split('.')[0], str(decision_tag))
    with open(osp.join(DATA_PATH, 'csv_results','{}_{}.csv'.format(exp, TAG)),'w') as fout:
        fout.write(rlt)


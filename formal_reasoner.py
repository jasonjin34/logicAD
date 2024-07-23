"""
This script takes a json file as input, convert the 'quasi'-formal output of LLM to well-formed formulae which are readable by prover9/mace4. Then it calls mace 4 for SAT-checking.
"""

import json
import os

ADDR_PROVER9 = '/Users/fengqihui/Desktop/Works/Prover/LADR-2009-11A/bin/prover9'
ADDR_MACE4 = '/Users/fengqihui/Desktop/Works/Prover/LADR-2009-11A/bin/mace4'
EXP_TAG = "breakfast_box_2207"

json_path = '/Users/fengqihui/Documents/GitHub/logicAD/datasets/formal/breakfast_box_v1.1.json'

with open(json_path, 'r') as jfile:
    data = json.load(jfile)

"""
    Example for converting one desc
"""
# print(data)
img = list(data.keys())[0]
img_tag = '_'.join(img.split('/')[-3:])
img_tag = img_tag.replace('.png', '.in')
desc = data[img]

# Processing
desc = desc.split('FORMULA')[-1]
desc = desc.strip('`')
desc = desc.strip(':')
desc = desc.strip(' ')
desc = desc.strip('\n')
desc = desc.replace('OR', '|')
desc = desc.replace('AND', '&')
desc = desc.replace('NOT', '-')

spec = "("
desc_list = desc.split('\n')
for desc_item in desc_list:
    if desc_item.strip(' ') != '':
        spec +='(' + desc_item.split('|')[0] + ')&'
spec = spec.strip('&') + ')'
# assump = desc.replace('\n', '.\n') + '.\n'
# goal = "(" + ")&(".join(desc.split('\n')) + ")."
rules = """left(orange,2).
left(apple,1).
right(cereal,irrel).
right(banana_chip, irrel).
right(almond, irrel).\n"""

rules += """all x ( all y (left(x,y)-> - (exists z (left(x,z) & z!=y)))).\nall x ( all y (right(x,y)-> - (exists z (left(x,z) & z!=y)))).\n"""

const_list = ['orange', 'apple', 'cereal','tangerine','granola','banana_chip', 'dried_banana_slice','almond','nut','0','1','2','3','irrel']
synom_list = [
    ['orange','mandarin','tangerine'],
    ['cereal','granola'],
    ['nut','almond'],
    ['banana_chip','dried_banana_slice']
    ]
# unique name assumption
una = ""
const_size = len(const_list)
for i1 in range(const_size):
    c1 = const_list[i1]
    for i2 in range(i1+1,const_size):
        c2 = const_list[i2]
        is_synom = False
        for synom_set in synom_list:
            if c1 in synom_set and c2 in synom_set:
                is_synom = True
                continue
        if not is_synom:
            una += "{}!={}.\n".format(c1,c2)

# synonymous axioms

sa = ""
for synom_set in synom_list:
    for i1 in range(len(synom_set)):
        for i2 in range(i1+1, len(synom_set)):
            sa += "{}={}.\n".format(synom_set[i1],synom_set[i2])

# functional axioms:
pred_list = ['left', 'right']
funcax = ""
for pred in pred_list:
    funcax += "(all x all y {}(x,y) -> (({}) & ({}))).\n".format(pred,'|'.join('x=' + c for c in const_list), ' | '.join('y=' + c for c in const_list))
# closed world assumption
cwa = ""
cwa += "(all x (x != orange & x != apple) -> left(x,0)).\n"
cwa += "(all x (x != cereal & x != nuts & x != banana_chip) -> right(x,0)).\n"

# print(una + funcax + sa + rules + cwa)

# print("#########")

# print('-({}).'.format(spec) )
# print(img)
# print(assump)
# print(goal)

prover_input = """
if(Prover9). % Options for Prover9
  assign(max_seconds, 10).
end_if.

if(Mace4).   % Options for Mace4
  assign(max_seconds, 10).
end_if.

formulas(assumptions).

{}

end_of_list.

formulas(goals).

{}

end_of_list.
""".format(una + funcax + sa + rules + cwa, '-({}).'.format(spec))

proofs_repo = "/Users/fengqihui/Documents/GitHub/logicAD/datasets/proofs"

# exp_tag = "breakfast_box_2207"

if os.path.isdir(proofs_repo):
    if not os.path.exists(proofs_repo + '/' + EXP_TAG):
        os.mkdir(proofs_repo + '/' + EXP_TAG)

prover_input_path = proofs_repo + '/' + EXP_TAG + '/' + img_tag
with open( prover_input_path, 'w+') as outfile:
    outfile.write(prover_input)

os.system("{} -f {} > {}".format(ADDR_PROVER9, prover_input_path, prover_input_path.replace('.in', '.out')))
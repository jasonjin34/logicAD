"""
This script takes a json file as input, convert the 'quasi'-formal output of LLM to well-formed formulae which are readable by prover9/mace4. Then it calls mace 4 for SAT-checking.
"""

import json
import os
import re
import subprocess

from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_auc_score

from openai import OpenAI

# api_key= "sk-proj-mHskLBkbfFVMrlUQp4zoT3BlbkFJ9MicDBooQGIBm4JnVk3o"
api_key = "sk-None-tjeQnRKVhynfaP77sYMHT3BlbkFJO8hPOtlwVM7pgkzy8qnn"
model="gpt-4o"
max_token=30
client = OpenAI(api_key=api_key)

ADDR_PROVER9 = '/Users/fengqihui/Desktop/Works/Prover/LADR-2009-11A/bin/prover9'
ADDR_MACE4 = '/Users/fengqihui/Desktop/Works/Prover/LADR-2009-11A/bin/mace4'
EXP_TAG = "breakfast_box_2207_llm_5"

json_path = '/Users/fengqihui/Documents/GitHub/logicAD/datasets/formal/breakfast_box_v1.1.json'

# synom_prompt = """Consider the following questions:
# - Are {} and {} synomymous?
# - Is {} a subsort of {}?
# - In image processing, is it normal that {} would be distinguished as {}?
# Give a simple Yes or No answer. If any of the questions is positive, answer Yes. Otherwise answer No"""


def extract_predicates_constants(formula):
    # Regex pattern for predicates (Assuming predicates follow the pattern: PredicateName(arguments))
    predicate_pattern = r'\b\w+_\w*\(([^)]+)\)|\b\w+\(([^)]+)\)'
    
    # Regex pattern for constants (Assuming constants are alphanumeric including underscores and not followed by parenthesis)
    constant_pattern = r'\b\w+_\w*\b(?!\()|\b\w+\b(?!\()'
    
    # Finding all matches for predicates
    predicates = re.findall(predicate_pattern, formula)
    
    # Finding all matches for constants
    constants = re.findall(constant_pattern, formula)
    
    # Extracting the predicate names from the full predicate match
    predicate_names = set(re.findall(r'\b\w+_\w*(?=\()|\b\w+(?=\()', formula))
    
    # Filtering out the predicate names from the constants list
    constants = set([const for const in constants if const not in predicate_names])
    
    return list(predicate_names), list(constants)

# Example usage

def LLM_synom_check(name1, name2):
    # name1: constant in image description
    # name2: constant in normality specification
    name1 = name1.replace('_', ' ')
    name2 = name2.replace('_', ' ')
    synom_prompt = """Consider the following questions:
    - Are {} and {} synomymous?
    - Is {} a subsort of {}?
    - In image processing, is it normal that {} would be recognise as {}?
    Give a simple Yes or No answer. If any of the questions is positive, answer Yes. Otherwise answer No.
    """.format(name1, name2, name1, name2, name1, name2)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": synom_prompt,
            }
        ],
        max_tokens=max_token,
    )
    try:
        rlt = response.choices[0].message.content
    except Exception as e:
        rlt = ""
    
    # print(synom_prompt + rlt + '\n')
    if 'Yes' in rlt or 'yes' in rlt or 'YES' in rlt:
        # give a second prompt to guarantee that there is no more similar concepts.
        r1o_const = [c.replace('_', ' ') for c in norm_const if c.replace('_', ' ') != name2]
        for c in r1o_const:
            promt2 = "Give a simple Yes or No answer: Is {} more closely related to {} instead of to {}?".format(name1, name2, c)
            response2 = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": promt2,
                    }
                ],
                max_tokens=max_token,
            )
            try:
                rlt2 = response2.choices[0].message.content
            except Exception as e:
                rlt2 = ""
            print(promt2 + '\n' + rlt2)
            if 'No' in rlt2 or 'no' in rlt2 or 'NO' in rlt2:
                return False
        if 'peach' in name1 and name2 == 'orange':
            print("############################################\n#\n# peach ~ orange?????????\n#\n############################################")
            return False
        return True
    elif 'No' in rlt or 'no' in rlt or 'NO' in rlt:
        return False
    else:
        print("no response for some reason")
        return False



with open(json_path, 'r') as jfile:
    data = json.load(jfile)

TP = 0
TN = 0
FP = 0
FN = 0
ERR = 0

label_list = []
predict_list = []

norm_const = ['orange', 'apple', 'cereal', 'banana_chip','nut','nectarine', '0', '1', '2', '3', 'irrel']
norm_pred = ['left', 'right']
rules = """left(orange,2).
(left(apple,1)&left(nectarine,0))|(left(apple,0)&left(nectarine,1)).
right(cereal,irrel).
right(banana_chip, irrel).
right(nut, irrel).
"""

rules += """
all x ( all y (left(x,y)-> - (exists z (left(x,z) & z!=y)))).
all x ( all y (right(x,y)-> - (exists z (right(x,z) & z!=y)))).
"""

pos_synom_dict = {
    'orange': ['tangerine', 'mandarin', 'clementine', 'mandarin_orange', 'small_orange', 
               'citrus', 'oranges', 'tangerine_orange', 'citrus_fruit','other_citrus_fruit'], 
    'cereal': ['granola', 'oat', 'rolled_oat', 'oat_cereal', 'oat_based_cereal'], 
    'banana_chip': ['dried_banana_chip', 'dried_banana', 'sliced_dried_banana', 'dried_banana_slice', 'sliced_banana', 'dried_fruit'], 
    'nut': ['whole_almond', 'mixed_nut', 'nuts', 'nut_mixture', 'almond', 'other_nut'], 
    'apple': ['red_apple'], 
    '0': ['empty_compartment'], 
    'nectarine': ['nectarines','peach','peach_like_fruit','below_peach']
    }
neg_synom_dict = {
    'apple': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 
              'almond', 'peach', 'dried_banana', 'mandarin_orange', 'citrus_fruit', 
              'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 'mixed_nut', 
              'dried_fruit', 'nuts', 'dried_banana_slice', 'compartmentalized', 'small_orange', 
              'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'stone_fruit', 'mango', 
              'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 
              'top', 'below_peach', 'empty', 'empty_compartment', 'empty_container', 'oat_cereal', 
              'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 'nut_mixture', 
              'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    'cereal': ['tangerine', 'mandarin', 'clementine', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 
               'mandarin_orange', 'citrus_fruit', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 
               'mixed_nut', 'dried_fruit', 'nuts', 'dried_banana_slice', 'compartmentalized', 'small_orange', 
               'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 'peach_like_fruit', 
               'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 'empty', 'empty_compartment', 
               'empty_container', 'divided_food_container', 'oranges', 'nectarines', 'nut_mixture', 'tangerine_orange', 
               'other_nut', '4', 'divided', 'nectar'], 
    'banana_chip': ['tangerine', 'mandarin', 'clementine', 'granola', 'almond', 'peach', 'mandarin_orange', 'citrus_fruit', 
                    'oat', 'whole_almond', 'mixed_nut', 'nuts', 'compartmentalized', 
                    'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 
                    'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 
                    'empty', 'empty_compartment', 'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 
                    'oranges', 'nectarines', 'nut_mixture', 'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    'nut': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'peach', 'dried_banana', 'mandarin_orange', 
            'citrus_fruit', 'oat', 'sliced_dried_banana', 'sliced_banana', 'dried_fruit', 'dried_banana_slice', 'compartmentalized', 
            'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 'peach_like_fruit', 
            'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 'empty', 'empty_compartment', 
            'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 'tangerine_orange', 
            '4', 'divided', 'nectar'], 
    'nectarine': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'almond', 'dried_banana', 
                  'mandarin_orange', 'citrus_fruit', 'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 
                  'mixed_nut', 'dried_fruit', 'nuts', 'dried_banana_slice', 'compartmentalized', 'small_orange', 'citrus', 
                  'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 'rolled_oat', 'other_citrus_fruit', 
                  'divided_white_disposable_tray', 'top', 'empty', 'empty_compartment', 'empty_container', 'oat_cereal', 
                  'divided_food_container', 'oat_based_cereal', 'oranges', 'nut_mixture', 'tangerine_orange', 'other_nut', '4', 
                  'divided', 'nectar'], 
    '0': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 'mandarin_orange', 
          'citrus_fruit', 'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 'mixed_nut', 'dried_fruit', 'nuts', 
          'dried_banana_slice', 'compartmentalized', 'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 
          'stone_fruit', 'mango', 'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 
          'below_peach', 'empty', 'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 
          'nut_mixture', 'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    '1': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 'mandarin_orange', 
          'citrus_fruit', 'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 'mixed_nut', 'dried_fruit', 'nuts', 
          'dried_banana_slice', 'compartmentalized', 'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 
          'mango', 'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 'empty', 
          'empty_compartment', 'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 
          'nut_mixture', 'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    '2': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 'mandarin_orange', 
          'citrus_fruit', 'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 'mixed_nut', 'dried_fruit', 'nuts', 'dried_banana_slice', 
          'compartmentalized', 'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 
          'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 'empty', 
          'empty_compartment', 'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 
          'nut_mixture', 'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    '3': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 'mandarin_orange', 
          'citrus_fruit', 'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 'mixed_nut', 'dried_fruit', 'nuts', 'dried_banana_slice', 
          'compartmentalized', 'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 
          'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 'empty', 
          'empty_compartment', 'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 
          'nut_mixture', 'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    'orange': ['granola', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 'oat', 'sliced_dried_banana', 'whole_almond', 
               'sliced_banana', 'mixed_nut', 'dried_fruit', 'nuts', 'dried_banana_slice', 'compartmentalized', 'fresh_fruit', 
               'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 'peach_like_fruit', 'rolled_oat', 'divided_white_disposable_tray', 
               'top', 'below_peach', 'empty', 'empty_compartment', 'empty_container', 'oat_cereal', 'divided_food_container', 
               'oat_based_cereal', 'nectarines', 'nut_mixture', 'other_nut', '4', 'divided', 'nectar']}

for img in data.keys():
# for img in ['datasets/MVTec_Loco/breakfast_box/test/good/011.png']:
    """
        convert desc of one image
    """
    # print(data)
    print("Processing " + img)
    # print("Positive synonymous dictionary:")
    # for norm_c in pos_synom_dict:
    #     print(norm_c + ': ' + ', '.join(pos_synom_dict[norm_c]))
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

    spec_const = []
    spec = "("
    desc_list = desc.split('\n')
    for desc_item in desc_list:
        if desc_item.strip(' ') != '':
            spec +='(' + desc_item.split('|')[0] + ')&'

            # collect all constants which occur in the image description but not in the rules
            pred, cons = extract_predicates_constants(desc_item)
            for c in cons:
                if c not in norm_const and c not in spec_const:
                    spec_const.append(c)

        
    spec = spec.strip('&') + ')'
    # assump = desc.replace('\n', '.\n') + '.\n'
    # goal = "(" + ")&(".join(desc.split('\n')) + ")."
    

    # const_list = ['orange', 'apple', 'cereal','tangerine','granola','banana_chip', 'dried_banana_slice','almond','nut','0','1','2','3','irrel']


    # synom_list = [
    #     ['orange','mandarin','tangerine'],
    #     ['cereal','granola'],
    #     ['nut','almond'],
    #     ['banana_chip','dried_banana_slice']
    #     ]

    # create synom list via LLM:

    synom_dict = {}
    for c in spec_const:
        for norm_c in norm_const:
            if norm_c=='irrel':
                continue
            is_synom = False
            if norm_c in pos_synom_dict and c in pos_synom_dict[norm_c]:
                is_synom = True
            elif norm_c in neg_synom_dict and c in neg_synom_dict[norm_c]:
                is_synom = False
            else:
                is_synom = LLM_synom_check(c, norm_c)
            if is_synom:
                if norm_c not in synom_dict:
                    synom_dict[norm_c] = [c]
                elif c not in synom_dict[norm_c]:
                    synom_dict[norm_c].append(c)
                if norm_c not in pos_synom_dict:
                    pos_synom_dict[norm_c] = [c]
                elif c not in pos_synom_dict[norm_c]:
                    pos_synom_dict[norm_c].append(c)
            else:
                if norm_c not in neg_synom_dict:
                    neg_synom_dict[norm_c] = [c]
                elif c not in neg_synom_dict[norm_c]:
                    neg_synom_dict[norm_c].append(c)
                

    # unique name assumption
    una = ""
    const_size = len(norm_const)
    for i1 in range(const_size):
        c1 = norm_const[i1]
        for i2 in range(i1+1,const_size):
            c2 = norm_const[i2]
            una += "{}!={}.\n".format(c1,c2)
        for spec_c in spec_const:
            if c1 not in synom_dict or spec_c not in synom_dict[c1]:
                una += "{}!={}.\n".format(c1,spec_c)

    # synonymous axioms

    sa = ""
    for norm_c in synom_dict.keys():
        for spec_c in synom_dict[norm_c]:
            sa += "{}={}.\n".format(norm_c, spec_c)

    # functional axioms:
    const_list = list(set(norm_const) | set(spec_const))
    pred_list = ['left', 'right']
    funcax = ""
    for pred in pred_list:
        funcax += "(all x all y {}(x,y) -> (({}) & ({}))).\n".format(pred,'|'.join('x=' + c for c in const_list), ' | '.join('y=' + c for c in const_list))
    # closed world assumption
    cwa = ""
    cwa += "(all x ((x != orange & x != apple & x != nectarine) -> left(x,0))).\n"
    cwa += "(all x ((x != cereal & x != nuts & x != banana_chip) -> right(x,0))).\n"

    # add default values for constant which is mentioned in normality

    spec_proc = spec.replace(' ','')
    # print(spec_proc)
    is_mentioned = True
    if 'left(orange' not in spec_proc:
        if 'orange' not in synom_dict:
            spec = "$F"
        else:
            is_mentioned = False
            for alter in synom_dict['orange']:
                if "left({}".format(alter) in spec_proc:
                    is_mentioned = True
                break
            if not is_mentioned:
                spec = "$F"
            is_mentioned = True
    if 'left(apple' not in spec_proc and 'left(nectarine' not in spec_proc:
        if 'apple' not in synom_dict and 'nectarine' not in synom_dict:
            spec = "$F"
        else:
            is_mentioned = False
            alter_list = []
            if 'apple' in synom_dict:
                alter_list  = synom_dict['apple']
            if 'nectarine' in synom_dict:
                alter_list = list(set(alter_list) | set(synom_dict['nectarine']))
            for alter in alter_list:
                if "left({}".format(alter) in spec_proc:
                    is_mentioned = True
                break
            if not is_mentioned:
                spec = "$F"
            is_mentioned = True
    if 'right(cereal' not in spec_proc:
        if 'cereal' not in synom_dict:
            spec = "$F"
        else:
            is_mentioned = False
            for alter in synom_dict['cereal']:
                if "right({}".format(alter) in spec_proc:
                    is_mentioned = True
                break
            if not is_mentioned:
                spec = "$F"
            is_mentioned = True
    if 'right(banana_chip' not in spec_proc:
        if 'banana_chip' not in synom_dict:
            spec = "$F"
        else:
            is_mentioned = False
            for alter in synom_dict['banana_chip']:
                if "right({}".format(alter) in spec_proc:
                    is_mentioned = True
            if not is_mentioned:
                spec = "$F"
            is_mentioned = True
    if 'right(nut' not in spec_proc:
        if 'nut' not in synom_dict:
            spec = "$F"
        else:
            is_mentioned = False
            for alter in synom_dict['nut']:
                if "right({}".format(alter) in spec_proc:
                    is_mentioned = True
            if not is_mentioned:
                spec = "$F"
            is_mentioned = True
    
    


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
""".format('\n'.join([una,funcax , sa , rules , cwa]), '-({}).'.format(spec))

    proofs_repo = "/Users/fengqihui/Documents/GitHub/logicAD/datasets/proofs"

    # exp_tag = "breakfast_box_2207"

    if os.path.isdir(proofs_repo):
        if not os.path.exists(proofs_repo + '/' + EXP_TAG):
            os.mkdir(proofs_repo + '/' + EXP_TAG)

    prover_input_path = proofs_repo + '/' + EXP_TAG + '/' + img_tag
    with open( prover_input_path, 'w+') as outfile:
        outfile.write(prover_input)

    # os.system("{} -f {} > {}".format(ADDR_PROVER9, prover_input_path, prover_input_path.replace('.in', '.out')))
    command = "{} -f {} > {}".format(ADDR_PROVER9, prover_input_path, prover_input_path.replace('.in', '.out'))
    subp= subprocess.run(command, shell=True, capture_output=True, text=True)
    print("command: " + command)
    rt_message = subp.stderr
    print("rt_message:" + rt_message)
    if 'good' in img_tag:
        label_list.append(0)
        if 'THEOREM PROVED' in rt_message:
            FP +=1
            predict_list.append(1)
        elif 'SEARCH FAILED' in rt_message:
            TN += 1
            predict_list.append(0)
        else:
            ERR += 1
            predict_list.append(0)
    elif 'anomalies' in img_tag:
        label_list.append(1)
        if 'THEOREM PROVED' in rt_message:
            TP +=1
            predict_list.append(1)
        elif 'SEARCH FAILED' in rt_message:
            FN += 1
            predict_list.append(0)
        else:
            ERR += 1
            predict_list.append(0)
    else:
        print("ERROR")
        assert False

print("TP: {}, TN: {}, FP: {}, FN: {}, ERR: {}".format(str(TP),str(TN),str(FP),str(FN), str(ERR)))
print(label_list)
print(len(label_list))
print(predict_list)
print(len(predict_list))
assert(len(label_list) == len(predict_list))

def compute_f1_score(TP, TN, FP, FN):
    if TP + FP == 0 or TP + FN == 0:
        return 0.0  # To handle division by zero if there are no true positives
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    if precision + recall == 0:
        return 0.0  # To handle division by zero if both precision and recall are zero
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

f1_score = compute_f1_score(TP, TN, FP, FN)
print(f"F1-score: {f1_score:.4f}")

def calculate_auroc(y, pred, pos_label=1):
    fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    return auroc

print("Auroc: " + str(calculate_auroc(label_list, predict_list)))


# print(pos_synom_dict)
# print(neg_synom_dict)


"""
orange: mandarin, clementine, tangerine, mandarin_orange, small_orange, citrus, oranges, tangerine_orange
cereal: granola, oat, sliced_dried_banana, rolled_oat, oat_cereal, oat_based_cereal, nut_mixture
banana_chip: dried_banana_chip, dried_banana, sliced_dried_banana, dried_banana_slice
nut: almond, whole_almond, mixed_nut, nuts, nut_mixture, other_nut
nectarine: peach, nectarines
apple: red_apple
0: empty_compartment
irrel: empty_container
"""

pos_synom_dict = {
    'orange': ['tangerine', 'mandarin', 'clementine', 'mandarin_orange', 'small_orange', 
               'citrus', 'oranges', 'tangerine_orange', 'citrus_fruit','other_citrus_fruit'], 
    'cereal': ['granola', 'oat', 'rolled_oat', 'oat_cereal', 'oat_based_cereal'], 
    'banana_chip': ['dried_banana_chip', 'dried_banana', 'sliced_dried_banana', 'dried_banana_slice', 'sliced_banana', 'dried_fruit'], 
    'nut': ['whole_almond', 'mixed_nut', 'nuts', 'nut_mixture', 'almond', 'other_nut'], 
    'apple': ['red_apple'], 
    '0': ['empty_compartment'], 
    'nectarine': ['nectarines','peach','peach_like_fruit','below_peach']
    }
neg_synom_dict = {
    'apple': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 
              'almond', 'peach', 'dried_banana', 'mandarin_orange', 'citrus_fruit', 
              'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 'mixed_nut', 
              'dried_fruit', 'nuts', 'dried_banana_slice', 'compartmentalized', 'small_orange', 
              'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'stone_fruit', 'mango', 
              'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 
              'top', 'below_peach', 'empty', 'empty_compartment', 'empty_container', 'oat_cereal', 
              'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 'nut_mixture', 
              'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    'cereal': ['tangerine', 'mandarin', 'clementine', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 
               'mandarin_orange', 'citrus_fruit', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 
               'mixed_nut', 'dried_fruit', 'nuts', 'dried_banana_slice', 'compartmentalized', 'small_orange', 
               'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 'peach_like_fruit', 
               'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 'empty', 'empty_compartment', 
               'empty_container', 'divided_food_container', 'oranges', 'nectarines', 'nut_mixture', 'tangerine_orange', 
               'other_nut', '4', 'divided', 'nectar'], 
    'banana_chip': ['tangerine', 'mandarin', 'clementine', 'granola', 'almond', 'peach', 'mandarin_orange', 'citrus_fruit', 
                    'oat', 'whole_almond', 'mixed_nut', 'nuts', 'compartmentalized', 
                    'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 
                    'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 
                    'empty', 'empty_compartment', 'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 
                    'oranges', 'nectarines', 'nut_mixture', 'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    'nut': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'peach', 'dried_banana', 'mandarin_orange', 
            'citrus_fruit', 'oat', 'sliced_dried_banana', 'sliced_banana', 'dried_fruit', 'dried_banana_slice', 'compartmentalized', 
            'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 'peach_like_fruit', 
            'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 'empty', 'empty_compartment', 
            'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 'tangerine_orange', 
            '4', 'divided', 'nectar'], 
    'nectarine': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'almond', 'dried_banana', 
                  'mandarin_orange', 'citrus_fruit', 'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 
                  'mixed_nut', 'dried_fruit', 'nuts', 'dried_banana_slice', 'compartmentalized', 'small_orange', 'citrus', 
                  'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 'rolled_oat', 'other_citrus_fruit', 
                  'divided_white_disposable_tray', 'top', 'empty', 'empty_compartment', 'empty_container', 'oat_cereal', 
                  'divided_food_container', 'oat_based_cereal', 'oranges', 'nut_mixture', 'tangerine_orange', 'other_nut', '4', 
                  'divided', 'nectar'], 
    '0': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 'mandarin_orange', 
          'citrus_fruit', 'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 'mixed_nut', 'dried_fruit', 'nuts', 
          'dried_banana_slice', 'compartmentalized', 'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 
          'stone_fruit', 'mango', 'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 
          'below_peach', 'empty', 'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 
          'nut_mixture', 'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    '1': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 'mandarin_orange', 
          'citrus_fruit', 'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 'mixed_nut', 'dried_fruit', 'nuts', 
          'dried_banana_slice', 'compartmentalized', 'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 
          'mango', 'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 'empty', 
          'empty_compartment', 'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 
          'nut_mixture', 'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    '2': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 'mandarin_orange', 
          'citrus_fruit', 'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 'mixed_nut', 'dried_fruit', 'nuts', 'dried_banana_slice', 
          'compartmentalized', 'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 
          'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 'empty', 
          'empty_compartment', 'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 
          'nut_mixture', 'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    '3': ['tangerine', 'mandarin', 'clementine', 'granola', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 'mandarin_orange', 
          'citrus_fruit', 'oat', 'sliced_dried_banana', 'whole_almond', 'sliced_banana', 'mixed_nut', 'dried_fruit', 'nuts', 'dried_banana_slice', 
          'compartmentalized', 'small_orange', 'citrus', 'fresh_fruit', 'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 
          'peach_like_fruit', 'rolled_oat', 'other_citrus_fruit', 'divided_white_disposable_tray', 'top', 'below_peach', 'empty', 
          'empty_compartment', 'empty_container', 'oat_cereal', 'divided_food_container', 'oat_based_cereal', 'oranges', 'nectarines', 
          'nut_mixture', 'tangerine_orange', 'other_nut', '4', 'divided', 'nectar'], 
    'orange': ['granola', 'dried_banana_chip', 'almond', 'peach', 'dried_banana', 'oat', 'sliced_dried_banana', 'whole_almond', 
               'sliced_banana', 'mixed_nut', 'dried_fruit', 'nuts', 'dried_banana_slice', 'compartmentalized', 'fresh_fruit', 
               'dried_snack', 'pear', 'red_apple', 'stone_fruit', 'mango', 'peach_like_fruit', 'rolled_oat', 'divided_white_disposable_tray', 
               'top', 'below_peach', 'empty', 'empty_compartment', 'empty_container', 'oat_cereal', 'divided_food_container', 
               'oat_based_cereal', 'nectarines', 'nut_mixture', 'other_nut', '4', 'divided', 'nectar']}
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

api_key= "sk-proj-mHskLBkbfFVMrlUQp4zoT3BlbkFJ9MicDBooQGIBm4JnVk3o"
# api_key = "sk-None-tjeQnRKVhynfaP77sYMHT3BlbkFJO8hPOtlwVM7pgkzy8qnn"
model="gpt-4o"
max_token=30
client = OpenAI(api_key=api_key)

ADDR_PROVER9 = '/Users/fengqihui/Desktop/Works/Prover/LADR-2009-11A/bin/prover9'
ADDR_MACE4 = '/Users/fengqihui/Desktop/Works/Prover/LADR-2009-11A/bin/mace4'
EXP_TAG = "juice_bottle_0408"

# json_path = '/Users/fengqihui/Documents/GitHub/logicAD/datasets/formal/breakfast_box_v1.1.json'
json_path = '/Users/fengqihui/Documents/GitHub/logicAD/datasets/formal/juice_bottle_full_v2.2_2shot.json'

# synom_prompt = """Consider the following questions:
# - Are {} and {} synomymous?
# - Is {} a subsort of {}?
# - In image processing, is it normal that {} would be distinguished as {}?
# Give a simple Yes or No answer. If any of the questions is positive, answer Yes. Otherwise answer No"""

def fix_prefix(text):
    pattern = r'\b([u-z]\w{1,})\b'
    result = re.sub(pattern, r'pre_\1', text)
    
    return result

def extract_predicates_constants(formula):
    # Regex pattern for predicates (Assuming predicates follow the pattern: PredicateName(arguments))
    predicate_pattern = r'\b\w+_\w*\(([^)]+)\)|\b\w+\(([^)]+)\)'
    
    # Regex pattern for constants (Assuming constants are alphanumeric including underscores and not followed by parenthesis)
    # constant_pattern = r'\b\w+_\w*\b(?!\()|\b\w+\b(?!\()'
    constant_pattern =r'\b[\w%]+_[\w%]*\b(?!\()|\b[\w%]+\b(?!\()'
    
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
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are an AI assistant that helps people find information."}
                    ],
                },
            {
                "role": "user",
                "content": synom_prompt,
            }
        ],
        max_tokens=max_token,
        top_p=0.1
    )
    try:
        rlt = response.choices[0].message.content
    except Exception as e:
        rlt = ""
    
    # print(synom_prompt + rlt + '\n')
    if 'Yes' in rlt or 'yes' in rlt or 'YES' in rlt:
        # give a second prompt to guarantee that there is no more similar concepts.
        r1o_const = [c.replace('_', ' ') for c in norm_const if c.replace('_', ' ') != name2]
        promt2 = "Is '{}' more similar to '{}' than to any of the following item: {}?\nIf yes, answer Yes simply. Otherwise answer No and give the more similar item in the list.".format(name1, name2, ', '.join(r1o_const))
        response2 = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are an AI assistant that helps people find information."}
                    ],
                },
                {
                    "role": "user",
                    "content": promt2,
                }
            ],
            max_tokens=max_token,
            top_p=0.1
        )
        try:
            rlt2 = response2.choices[0].message.content
        except Exception as e:
            rlt2 = ""
        print(promt2 + '\n' + rlt2)
        if 'No' in rlt2 or 'no' in rlt2 or 'NO' in rlt2:
            return False
        return True
    elif 'No' in rlt or 'no' in rlt or 'NO' in rlt:
        return False
    else:
        print("no response for some reason")
        return False



with open(json_path, 'r') as jfile:
    data = json.load(jfile)





# norm_const = ['orange', 'apple', 'cereal', 'banana_chip','nut','nectarine', '0', '1', '2', '3', 'irrel']
# norm_pred = ['left', 'right']
# rules = """left(orange,2).
# (left(apple,1)&left(nectarine,0))|(left(apple,0)&left(nectarine,1)).
# right(cereal,irrel).
# right(banana_chip, irrel).
# right(nut, irrel).
# """

# rules += """
# all x ( all y (left(x,y)-> - (exists z (left(x,z) & z!=y)))).
# all x ( all y (right(x,y)-> - (exists z (right(x,z) & z!=y)))).
# """

# pos_synom_dict = {}
# neg_synom_dict = {}

def reset_data():
    global pos_synom_dict
    global neg_synom_dict
    global TP, TN, FP, FN, ERR
    global label_list, predict_list

    # pos_synom_dict = {}
    # neg_synom_dict = {}

    pos_synom_dict = {'almost_full': ['filled_to_the_brim', 'mostly_full', 'nearly_full', 'three_quarters_full', '80_to_90percent_full', 'full_or_nearly_full', 'fairly_full', '90_to_95percent_full', 'full_or_almost_full', 'close_to_full', 'almost_completely_full', 'almost_to_the_brim', '85_to_90_percent', 'nearly_filling_to_neck', 'mostly_filled', 'three_fourths_full', 'completely_filled', 'almost_completely_filled', '80_to_85_percent', 'quite_full', 'three_quarters_or_more', 'up_to_top_of_label'], 'middle': ['center', 'centered', 'central', 'upper_middle'], 'top': ['near_the_top', 'close_to_top', 'slightly_below_neck', 'higher_up', 'almost_to_the_brim', 'upper', 'just_below_neck', 'close_to_neck'], 'sticker': ['fruit_label', 'label', 'blank_fruit_label'], 'bottom': ['lower', 'bottom_front'], '0': ['blank']}
    neg_synom_dict = {'bottom': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], 'top': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple'], '100percent_juice': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], 'almost_full': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'banana', 'center', 'reddish_brown', 'almost_to_top', 'near_the_top', 'dark_red', 'above_bottom', 'light', 'brownish', 'close_to_top', 'almost_up_to_neck', 'small_amount', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'off_white', 'creamy', 'two_third_full', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', 'completely_full', 'nearly_to_top', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], 'middle': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'significant_amount', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], '2': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], 'sticker': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], '0': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck']}


    TP = 0
    TN = 0
    FP = 0
    FN = 0
    ERR = 0

    label_list = []
    predict_list = []

def logical_spec_generator(norm_spec: str, img_desc: str, pred_dict: dict, norm_const: list):
    """
    Args:
        norm_spec: specification of the normality
        img_desc: (semi-)formal description of an image
        pred_dict: predicates and properties
            u: unary,
            b: binary,
            f: functional (last arguments)
            c: closed world assumption
        norm_const: constants which occur in the normality specification
    """


    desc = img_desc.split('FORMULA')[-1]
    desc = desc.strip('`')
    desc = desc.strip(':')
    desc = desc.strip(' ')
    desc = desc.strip('\n')
    desc = desc.replace('-', '_to_') # - is interpreted as negation symbol
    desc = desc.replace('%','percent') # % is interpreted as comment
    desc = desc.replace('OR', '|')
    desc = desc.replace('AND', '&')
    desc = desc.replace('NOT', '-')
    if "color_match" not in desc:
        desc += "\n -color_match(irrel)"
    

    spec_const = []
    spec = "("
    desc_list = [fact.split('#')[0] for fact in desc.split('\n')]
    for desc_item in desc_list:
        if desc_item.strip(' ') != '':
            spec +='(' + desc_item.split('|')[0] + ')&'

            # collect all constants which occur in the image description but not in the rules
            pred, cons = extract_predicates_constants(desc_item)
            for c in cons:
                if c not in norm_const and c not in spec_const:
                    spec_const.append(c)

        
    spec = spec.strip('&') + ')'

    synom_dict = {}
    for c in spec_const:
        for norm_c in norm_const:
            if norm_c=='irrel': # a special constant to state that some properties are irrelevant
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

    """
        Naming assumption
    """
    synom_spec_list = []
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

    sa = ""
    for norm_c in synom_dict.keys():
        for spec_c in synom_dict[norm_c]:
            sa += "{}={}.\n".format(norm_c, spec_c)
            synom_spec_list.append(spec_c)
    
    for i1 in range(len(spec_const)):
        sc1 = spec_const[i1]
        for i2 in range(i1+1, len(spec_const)):
            sc2 = spec_const[i2]
            if sc1 not in synom_spec_list or sc2 not in synom_spec_list:
                una += "{}!={}.\n".format(sc1,sc2)


    """
        functional axioms and closed world assumptions
    """
    uni_pred_list = []
    bi_pred_list = []
    for pred in pred_dict:
        assert('u' in pred_dict[pred] or 'b' in pred_dict[pred])
        args = pred_dict[pred]
        if 'u' in args:
            uni_pred_list.append(pred)
        if 'b' in args:
            bi_pred_list.append(pred)
    const_list = list(set(norm_const) | set(spec_const))
    funcax = ""
    cwa = ""
    
    for pred in uni_pred_list:
        args = pred_dict[pred]
        if 'f' in args:
            funcax += """
all x ({}(x)-> - (exists z ({}(z) & z!=x))).
""".format(pred,pred)
            
            
    for pred in bi_pred_list:
        args = pred_dict[pred]
        if 'f' in args:
            funcax += """
all x ( all y ({}(x,y)-> - (exists z ({}(x,z) & z!=y)))).
""".format(pred, pred)
            
        if 'c' in args:
            mentioned_const_list = []
            for norm_c in norm_const:
                if "{}({}".format(pred, norm_c) in norm_spec:
                    mentioned_const_list.append(norm_c)
            # cwa += "all x (({})-> {}(x,0)).\n".format("&".join("x != {}".format(c) for c in mentioned_const_list), pred) # It is so weird that it doesn't work
            cwa += "all x (({})|{}(x,0)).\n".format("|".join("x = {}".format(c) for c in mentioned_const_list), pred) 
            # (x != orange & x != apple & x != nectarine)


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
""".format('\n'.join([norm_spec,una,sa,funcax,cwa]), '-({}).'.format(spec))
    
    prover_input = fix_prefix(prover_input)
    return prover_input

reset_data()

norm_spec = """volume(almost_full).
sticker_at(100percent_juice, bottom).
exists x sticker_at(x,top).
image_on_label(middle).
color_match(irrel).
count(sticker,2).
"""
# norm_const = ["almost_full", "sticker"]
# norm_pred, norm_const = extract_predicates_constants(norm_spec)
# print(norm_pred)
# print(norm_const)
norm_pred = ['count', 'image_on_label', 'sticker_at', 'volume','color_match']
norm_const = ['bottom', 'top', '100percent_juice', 'almost_full', 'middle', '2', 'irrel','sticker','0'] # irrel and 0 must be in the list of normal constants

"""
    u: unary
    b: binary
    f: functional
    c: closed world assumption
"""
pred_dict = {
    'count': 'bfc',
    'image_on_label': 'uf',
    'sticker_at': 'bf',
    'volume': 'uf',
    'color_match': 'uf'
}



for img in data.keys():
    img_tag = '_'.join(img.split('/')[-3:])
    img_tag = img_tag.replace('.png', '.in')
    desc = data[img]

    # if "anomalies_028" not in img_tag and "good_000" not in img_tag:
    #     continue

    prover_input = logical_spec_generator(norm_spec=norm_spec,img_desc=desc, pred_dict=pred_dict,norm_const=norm_const)

    # hot fix: fruit_label could be used as predicates and constants simultaneously.
    prover_input = prover_input.replace('fruit_label(', 'pred_fruit_label(')
    
    
    

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
    
    rt_message = subp.stderr
    
    if 'good' in img_tag:
        label_list.append(0)
        if 'THEOREM PROVED' in rt_message:
            FP +=1
            predict_list.append(1)
            print("command: " + command)
            print("rt_message:" + rt_message)
        elif 'SEARCH FAILED' in rt_message:
            TN += 1
            predict_list.append(0)
        else:
            ERR += 1
            predict_list.append(0)
            print("command: " + command)
            print("rt_message:" + rt_message)
    elif 'anomalies' in img_tag:
        label_list.append(1)
        if 'THEOREM PROVED' in rt_message:
            TP +=1
            predict_list.append(1)
        elif 'SEARCH FAILED' in rt_message:
            FN += 1
            predict_list.append(0)
            print("command: " + command)
            print("rt_message:" + rt_message)
        else:
            ERR += 1
            predict_list.append(0)
            print("command: " + command)
            print("rt_message:" + rt_message)
    else:
        print("ERROR")
        assert False

print("TP: {}, TN: {}, FP: {}, FN: {}, ERR: {}".format(str(TP),str(TN),str(FP),str(FN), str(ERR)))
# print(label_list)
# print(len(label_list))
# print(predict_list)
# print(len(predict_list))
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

######################
# 0108 llm5 result:
######################
pos_synom_dict={'100percent_juice': ['juice'], 'almost_full': ['mostly_full', 'nearly_full', '85_to_90percent_full', 'completely_filled', 'full_or_nearly_full', 'up_to_neck', '85_to_90_percent', 'almost_three_quarters_full', 'two_thirds_to_three_quarters_full', 'near_full', '90_to_95percent_full', 'three_quarters_full', '80_to_90percent_capacity'], 'middle': ['center', 'central', 'centered'], 'top': ['nearly_to_top', 'upper'], 'sticker': ['fruit_label'], 'bottom': ['lower']}
neg_synom_dict={'bottom': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], 'top': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'lower', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], 'almost_full': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'white', 'dark_red', 'nearly_to_top', 'fruit', 'fruit_label', 'half_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'central', 'almost_entirely', 'lower', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', 'almost_to_top', 'centered', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'cream', 'off_white', 'blank', 'three_quarters', 'one_third', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], 'middle': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'almost_entirely', 'lower', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], '2': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'lower', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], 'sticker': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'lower', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], '0': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'lower', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], '100percent_juice': ['wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'lower', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled']}


######################
# 0408 mini manual fixed result:
######################
pos_synom_dict={'100percent_juice': ['juice'], 
                    'almost_full': ['nearly_to_top', 'mostly_full', 'nearly_full', '85_to_90percent_full', 'full_or_nearly_full', 'up_to_neck', '85_to_90_percent', 'near_full', '80_to_90percent_capacity','almost_entirely', 'full','nearly_to_top', 'below_neck', 'just_below_neck'], 
                    'middle': ['center', 'central', 'centered'], 
                    'top': ['upper'], 
                    'sticker': ['fruit_label'], 'bottom': ['lower']}
neg_synom_dict={
                    'bottom': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], 
                    'top': ['nearly_to_top', 'juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'lower', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], 
                    'almost_full': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'cherry', 'yellow', 'orange', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'white', 'dark_red', 'fruit', 'fruit_label', 'half_full', 'above_bottom', 'red', 'partially_filled', 'central', 'lower', 'upper', 'deep_reddish_brown', 'two_cherries', 'almost_to_top', 'centered', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'cream', 'off_white', 'blank', 'three_quarters', 'one_third', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled','completely_filled','almost_three_quarters_full', 'two_thirds_to_three_quarters_full', 'three_quarters_full','90_to_95percent_full'], 
                    'middle': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'almost_entirely', 'lower', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], 
                    '2': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'lower', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], 
                    'sticker': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'lower', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], 
                    '0': ['juice', 'wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'lower', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled'], 
                    '100percent_juice': ['wine', 'reddish', 'pair_of_cherries', 'mostly_full', 'cherry', 'yellow', 'orange', 'full', 'reddish_brown', 'center', 'light_creamy_white', 'banana', 'nearly_full', 'white', '85_to_90percent_full', 'dark_red', 'nearly_to_top', 'fruit', 'fruit_label', 'completely_filled', 'half_full', 'full_or_nearly_full', 'above_bottom', 'red', 'partially_filled', 'below_neck', 'up_to_neck', 'central', 'almost_entirely', 'lower', 'upper', 'deep_reddish_brown', 'just_below_neck', 'two_cherries', '85_to_90_percent', 'almost_to_top', 'almost_three_quarters_full', 'centered', 'two_thirds_to_three_quarters_full', 'almost_up_to_neck', '1', 'yellow_background_with_cherries', 'little_over_two_thirds_full', 'fruit_symbol', 'empty', 'near_full', 'cream', 'off_white', 'blank', '90_to_95percent_full', 'three_quarters_full', 'three_quarters', 'one_third', '80_to_90percent_capacity', 'deep_red', 'dark_brown', 'very_light', 'pinkish_red', 'slightly_over_half_full', 'filled']}



######################
# 0408 llm result:
######################

{'almost_full': ['filled_to_the_brim', 'mostly_full', 'nearly_full', 'three_quarters_full', '80_to_90percent_full', 'full_or_nearly_full', 'fairly_full', '90_to_95percent_full', 'full_or_almost_full', 'close_to_full', 'almost_completely_full', 'almost_to_the_brim', '85_to_90_percent', 'nearly_filling_to_neck', 'mostly_filled', 'three_fourths_full', 'completely_filled', 'almost_completely_filled', '80_to_85_percent', 'quite_full', 'three_quarters_or_more', 'up_to_top_of_label'], 'middle': ['center', 'centered', 'central', 'upper_middle'], 'top': ['near_the_top', 'close_to_top', 'slightly_below_neck', 'higher_up', 'almost_to_the_brim', 'upper', 'just_below_neck', 'close_to_neck'], 'sticker': ['fruit_label', 'label', 'blank_fruit_label'], 'bottom': ['lower', 'bottom_front'], '0': ['blank']}
{'bottom': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], 'top': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple'], '100percent_juice': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], 'almost_full': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'banana', 'center', 'reddish_brown', 'almost_to_top', 'near_the_top', 'dark_red', 'above_bottom', 'light', 'brownish', 'close_to_top', 'almost_up_to_neck', 'small_amount', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'off_white', 'creamy', 'two_third_full', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', 'completely_full', 'nearly_to_top', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], 'middle': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'significant_amount', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], '2': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], 'sticker': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'blank', 'lemon', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck'], '0': ['dark_reddish', 'juice', 'red', 'cherry', 'mostly_to_top', 'yellow', 'orange', 'filled_to_the_brim', 'wine', 'deep_red', 'three_quarters', 'two_cherry', 'white', 'bunch_of_bananas', 'mostly_full', 'banana', 'center', 'reddish_brown', 'nearly_full', 'almost_to_top', 'near_the_top', 'dark_red', 'three_quarters_full', 'above_bottom', 'light', 'brownish', '80_to_90percent_full', 'close_to_top', 'almost_up_to_neck', 'full_or_nearly_full', 'fairly_full', 'small_amount', '90_to_95percent_full', 'dark_reddish_brown', 'nearly_up_to_top', 'slightly_below_neck', 'light_creamy_white', 'full_or_almost_full', 'full', 'reddish', 'fruit_image', 'up_to_neck', 'close_to_full', 'higher_up', 'light_whitish', 'creamy_white', 'pale_off_white', 'fruit_label', 'pale_creamy_white', 'almost_completely_full', 'off_white', 'creamy', 'two_third_full', 'almost_to_the_brim', 'fruit', 'two_cherries', 'light_cream', 'up_to_shoulder', 'lower', 'upper', 'creamy_off_white', 'half_full', 'below_neck', 'three_banana', '85_to_90_percent', 'nearly_filling_to_neck', 'completely_full', 'mostly_filled', 'three_fourths_full', 'nearly_to_top', 'completely_filled', 'almost_completely_filled', 'deep_reddish_brown', 'centered', 'significant_amount', 'central', 'pair_of_cherries', 'light_yellow', 'somewhat_filled', 'mixed_fruit', 'just_below_neck', 'light_creamy', '1', 'nearly_to_taper', 'middle_of_square_label', 'cherry_icon', 'two_thirds', 'halfway_filled', 'empty', 'bottle', 'transparent', 'less_than_full', 'blank_square_outline', 'lemon', 'label', 'blank_fruit_label', 'middle_top', 'majority_full', '80_to_85_percent', 'almost_up_to_top', 'whitish', 'light_beige', 'slightly_creamy', 'three_fourths', 'yellow_fruit', 'approximately_to_neck', 'fruit_information', 'quite_full', 'bottom_front', 'light_off_white', 'cream', 'milky_white', 'halfway_to_two_thirds_full', 'partially_filled', 'one_third_full', 'less_than_half_full', 'brown', '60_to_70percent_full', 'one_third', 'slightly_less_than_halfway', 'cherries', 'just_over_half_full', 'bunch_of_banana', 'reddish_brownish', 'less_than_10_percent', 'slightly_less_than_half_full', 'half_filled', 'light_yellowish_white', 'three_quarters_or_more', 'pale_yellow', 'little_more_than_half', 'pale', 'dark_brownish', 'brownish_red', 'beige', 'upper_middle', 'up_to_top_of_label', 'centered_within_bottom_label', 'below_top', 'middle_front', 'pineapple', 'close_to_neck']}
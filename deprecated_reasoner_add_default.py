import os
origin_path = "/Users/fengqihui/Documents/GitHub/logicAD/datasets/proofs/breakfast_box_2207_llm_2"
new_path = "/Users/fengqihui/Documents/GitHub/logicAD/datasets/proofs/breakfast_box_2207_llm_2_add_default"

if not os.path.exists(new_path):
    os.mkdir(new_path)

# for file in os.path 
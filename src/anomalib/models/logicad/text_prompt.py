"""Logic-AD Text Prompt Module"""

# TODO:
# as for now the text prompt is class dependent we can eventually make it more general

TEXT_EXTRACTOR_PROMPTS = {
    # loco category
    "breakfast_box": "what is on the left side of image? and what is on the right side of image?",
    "juice_bottle": "what is color of the juice? what is the fruit? color of juice should match with fruit (red, wine color for cherry, white for banana and yellow for orange), how much juice in the bottle? are there two sticker, one with 100% juice on the bottom and the other with the fruit label on the top and fruit is located in the middle of the label?",
    "pushpins": "how many pushpins are there? give the answer as the following format: {pushpins: number of pushpins}",
    "screw_bag": "how many bolts, washers, and nuts? describe the length of the bolts using the longer bolt as reference (1/4, 1/2, 3/4, 1)",
    "splicing_connectors": "Answer this question if the image contain only one block of connectors: where is the vertical position of the cable (use top, middle or bottom of the connectors for description)?. Answer these questions if the image contains seperate connector blocks: How many connectors are there? how many cables are there? is the cable broken or not? is the connector has the same size?",
    # mvtec category
    "cable": "is there any flaw, abnormal, cut, untwisted or splayed out wird in the image, yes or no with short reason? do the inner insulation have completely color?",
    "pill": "is there any color other then sparse wine specks? do all specks have the same color? does the pill has crack or noticeable broken? is the pill has print FF flawlessly? pill is defect if one of the question is yes",
    "capsule": "does the capsule image contain scratch? does all parts intact or flawless? does the 500 print intact?",
    "screw": "does the screw contains the following anomalies, broken thread side, flaw thread top, scratch and broken front",
    "bottle": "is there any crack or broken part? is the center area uniformly dark? is there any object inside of the bottle? if one of the question is yes, then the bottle is defect", 
    "transistor": "is the transistor case flawless? is the transistors properly mounted?",
    "toothbrush": "is the bristels have similar length, not particular too long or too short? is there any empty whole in the toothbrush? is there any abnormal objects in the toothbrush?",
    "hazelnut": "is hazelnut broken? is the hazelnut shell intact? is the hazelnut shell flawless? is the hazelnut has print or mark?",
    "metal_nut": "does the metal nut contain any deep large scratch? are all lobes intact? is the metal contain color stain?",
    "zipper": "Is there any flaw, abnormality, cut, broken fabric in the image, yes or no? any splitting, unclosed, squeezed teeth in the zipper?",
    "wood": "is the wood flawless if there is no water mark, no scratch, no crack, no cut, no hole, give me a short reason",
    "carpet": "is the carpet has any scratch or cut? is the carpet has any color stain? is the carpet has metal contamination?",
    "leather": "is the leather flawless? (water mark is also consider as defect or flaw)",
    "tile": "is the tile flawless or not? (only crack, scratch, color stain, roughness surface, liquid stain or piece of glue tape , grey stain, are consider as defect or flaw)",
    "grid": "is the grid flawless or not? (only water mark, scratch, bent, broken and additional object are consider as defect or flaw)",
    # visa category
    "candle": "is the candle flawless?",
}

TEXT_SUMMATION_PROMPTS = {
    "breakfast_box": "number of objects",
    "juice_bottle": "color matching: {yes, or no, if there is no sticker then no}, juice status: {how full is the bottle},  top sticker: {correct if the fruit match with juice}, bottom sticker: {correct if there is word 100% juice}",
    "pushpins": "number of pushpin per patch: {[list of number of pushpins]}, same patches: {yes or no, no if two patches are different}",
    "screw_bag": "number of objects, length of bolts",
    "splicing_connectors": "connector: {nummber of connectors bl, size}, cable: {nummber of cables, broken or not?}, patchs: {same position: (answer it with yes or not)?}",
    # mvtec category
    "cable":  "defect: {yes or no, defect reason}, three unique color: {yes or no}",
    "pill":  "defect: {yes or no}, reason: {short reason}",
    "capsule":  "is the object intact, flawless or perfect?",
    "screw":  "flaw: {extend of flaw}",
    "bottle": "crack: {yes or no}, center: {uniformlly black or not}, object: {yes or no}",
    "transistor": "condition: ?",
    "toothbrush": "bristels: {similar length or not}, empty hole: {yes or no}, abnormal object: {yes or no}",
    "hazelnut": "is hazelnut broken, is hazelnut has cut? is hazelnut has print or mark?",
    "metal_nut": "scratch: {yes or no}, intact lobes?: {yes or no}, broken : {yes or no}, color stain: {yes or no}",
    'zipper': "defact: {yes or no}, reason: {short reason}",
    "wood": "flaw: {yes or no}, reason: ",
    "carpet": "defect: {yes or no}, reason: {short reason}",
    "leather": "defect: {yes or no}, reason: {short reason}",
    "tile": "defect: {yes or no}, reason: {short reason}",
    "grid": "defect: {yes or no}, reason: {short reason}",
    # visa category
    "candle": "defect: {yes or no}, reason: {short reason}",
}


TEXT_EXTRACTOR_PROMPTS_SA = {
    # loco category
    "breakfast_box": "any abnormal object in the image which does not belong to breakfast box (only with fruit, granola, banana chip, almond ? any broken object (banana chip and almod can not be in very tiny pieces)? only give short answer",
    "juice_bottle": "any object inside the bottle other than juice? is the juice color match with label (yellow for orange, white for banana, wine red for cherry? are all label intact? is the fruit picture correct (orange with one green leaf or two cherry or bananas)?",
    "pushpins": "is the pushpin broken or damaged? if yes, give a short reason. is the compartment contain other object other than pushpin?",
    "screw_bag": "how many bolts, washers, and nuts? describe the length of the bolts using the longer bolt as reference (1/4, 1/2, 3/4, 1)"
}

TEXT_SUMMATION_PROMPTS_SA = {
    "breakfast_box": "abnormal object: {yes or no}, broken object: {yes or no}",
    "juice_bottle": "abnormal object inside bottle: {yes or no}, juice color: {correct}, label: {intact or not}, fruit picture: {correct or not}",
    "pushpins": "damage or broken: {yes or no}, reason: {short reason if no}, other object: {yes or no}",
    "screw_bag": "number of objects, length of bolts",
    "splicing_connectors": "connector: {nummber of connectors bl, size}, cable: {nummber of cables, broken or not?}, patchs: {same position: (answer it with yes or not)?}"
}
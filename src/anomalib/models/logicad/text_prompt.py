"""Logic-AD Text Prompt Module"""

# TODO:
# as for now the text prompt is class dependent we can eventually make it more general

TEXT_EXTRACTOR_PROMPTS = {
    # loco category
    "breakfast_box": "what is on the left side of image? and what is on the right side of image?",
    # "juice_bottle": "Describe this juice bottle's characteristics, is the bottle full? locations of label and is objects",
    "juice_bottle": "what is color of the juice? what is the fruit? color of juice should match with fruit (red, wine color for cherry, white for banana and yellow for orange), how much juice in the bottle? are there two sticker, one with 100% juice on the bottom and the other with the fruit label on the top and fruit is located in the middle of the label?",
    "pushpins": "is there any pushpins in the image?, yes or no",
    "screw_bag": "how many bolts, washers, and nuts? describe the length of the bolts using the longer bolt as reference (1/4, 1/2, 3/4, 1)",
    "splicing_connectors": "Answer this question if the image contain only one connector: which slot is the cable connected to (top one is the first one)?. Answer these questions if the image contain more than one connector: How many connectors are there? how many cables are there? is the cable broken or not? is the connector has the same size?",
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
    # "juice_bottle": "location of objects, and their characteristics",
    "juice_bottle": "color matching: {yes, or no, if there is no sticker then no}, juice status: {how full is the bottle},  top sticker: {correct if the fruit match with juice}, bottom sticker: {correct if there is 100% juice states therer}",
    "pushpins": "pushpin: {yes or no}",
    "screw_bag": "number of objects, length of bolts",
    "splicing_connectors": "connector: {nummber of connectors}, cable: {nummber of cables, broken or not?}, patchs: {is both cable connected to the same slot on each side}",
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

TEXT_NUM_OBJECTS_PROMPTS = {
    "pushpins": {
        "number": 15,
        "prompt": "the total number of pushpins is not correct"
    }
}

PROMPT0 = (
    "Given the description of an image, please output a formal specification as a set of (propositional) formulae. "
)

RULE = """Some syntactical rules are to be followed:\n
1. Each line consists of only one piece of fact.\n
2. Predicates are named in terms of properties such as location, color, size etc. Connect words with underline. Use lowercases only.\n
3. Objects and quantities are given as arguments of the predicates (use * if the object is uncountable or the number is irrelevant). Connect words with underline and always use singular form\n
4. Logical connectives such as $AND$, $OR$ $NOT$ might be used.
5. Description and output should be given in form 'Text: (Description of the image)\nFormulae: (a set of logical formulae)'"""
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
    "cable":  "is the image flawless? if the following conditionals are reach, then the image is flawness or defect?, (bent wire, cable with less than 3 color pattern, cut_inner_insulation, missing cable, scratch), give me only short answer and reason",
    "pill": "is the pill flawless? if all the following conditions need are reach, no scratch, contanimination, no yellow spot, no crack, no fault print",
    "capsule": "does the capsule image contain scratch? does all parts intact or flawless? does the 500 print intact?",
    "screw": "does the screw contains the following anomalies, broken thread side, flaw thread top, scratch and broken front"
}

TEXT_SUMMATION_PROMPTS = {
    "breakfast_box": "number of objects",
    # "juice_bottle": "location of objects, and their characteristics",
    "juice_bottle": "color matching: {yes, or no, if there is no sticker then no}, juice status: {how full is the bottle},  top sticker: {correct if the fruit match with juice}, bottom sticker: {correct if there is 100% juice states therer}",
    "pushpins": "pushpin: {yes or no}",
    "screw_bag": "number of objects, length of bolts",
    "splicing_connectors": "connector: {nummber of connectors}, cable: {nummber of cables, broken or not?}, patchs: {is both cable connected to the same slot on each side}",
    # mvtec category
    "cable":  "is the object intact, flawless or perfect?",
    "pill":  "conditions: true or false",
    "capsule":  "is the object intact, flawless or perfect?",
    "screw":  "flaw: {extend of flaw}",
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
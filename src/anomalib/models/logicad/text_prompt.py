"""Logic-AD Text Prompt Module"""

# TODO:
# as for now the text prompt is class dependent we can eventually make it more general

TEXT_EXTRACTOR_PROMPTS = {
    "breakfast_box": "what is on the left side of image? and what is on the right side of image?",
    "juice_bottle": "Describe this juice bottle's characteristics, is the bottle full? locations of label and is objects",
    "pushpins": "are different pushpins divided by plastic wall?, or only contain one pushpin?",
    "screw_bag": "What are inside the bag?",
    "splicing_connectors": "Describe the splicing connectors and cables in the image. Is the connector symmetric? Is the cable symmetric? Is the cable broken? connector should be far away from each other"
}

TEXT_SUMMATION_PROMPTS = {
    "breakfast_box": "number of object: {equal to 15 or not}, patches: {summary of unique patches}",
    "juice_bottle": "location of objects, and their characteristics",
    "pushpins": "number of pushpins, and patches characteristics",
    "screw_bag": "number of objects",
    "splicing_connectors": "connector: {symmetric or not, number of connectors}, cable: {nummber of cables, symmetric:, broken or not?, length: longer than connector}",
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
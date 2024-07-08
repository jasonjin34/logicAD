"""Logic-AD Text Prompt Module"""

# TODO:
# as for now the text prompt is class dependent we can eventually make it more general

TEXT_EXTRACTOR_PROMPTS = {
    "breakfast_box": "what is on the left side of image? and what is on the right side of image?",
    "juice_bottle": "Describe this juice bottle's characteristics, is the bottle full? locations of label and is objects",
    "pushpins": None,
    "screw_bag": None,
    "splicing_connectors": None,
}

TEXT_SUMMATION_PROMPTS = {
    "breakfast_box": "number of objects, and their location",
    "juice_bottle": "location of objects, and their characteristics",
    "pushpins": None,
    "screw_bag": None,
    "splicing_connectors": None,
}

"""Logic-AD Text Prompt Module"""

# TODO:
# as for now the text prompt is class dependent we can eventually make it more general

TEXT_EXTRACTOR_PROMPTS = {
    "breakfast_box": "what is on the left side of image? and what is on the right side of image?",
    "juice_bottle": None,
    "pushpins": None,
    "screw_bag": None,
    "splicing_connectors": None,
}

TEXT_SUMMATION_PROMPTS = {
    "breakfast_box": "location: object ...",
    "juice_bottle": None,
    "pushpins": None,
    "screw_bag": None,
    "splicing_connectors": None,
}

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

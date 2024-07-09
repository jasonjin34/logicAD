"""Logic-AD Text Prompt Module"""

# TODO:
# as for now the text prompt is class dependent we can eventually make it more general

TEXT_EXTRACTOR_PROMPTS = {
    "breakfast_box": "what is on the left side of image? and what is on the right side of image?",
    "juice_bottle": "Describe this juice bottle's characteristics, is the bottle full? locations of label and is objects",
    "pushpins": None,
    "screw_bag": "What are inside the bag?",
    "splicing_connectors": "Describe the splicing connectors and cables in the image. Is the connector symmetric? Is the cable symmetric? Is the cable broken? connector should be far away from each other"
}

TEXT_SUMMATION_PROMPTS = {
    "breakfast_box": "number of objects, and their location",
    "juice_bottle": "location of objects, and their characteristics",
    "pushpins": None,
    "screw_bag": "number of objects",
    "splicing_connectors": "connector: {symmetric or not, number of connectors}, cable: {nummber of cables, symmetric:, broken or not?, length: longer than connector}",
}

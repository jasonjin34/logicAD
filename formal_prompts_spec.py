"""
    The prompts and logical specification for all datasets
"""

PROMPT0 = (
    "Given the description of an image, please output a formal specification as a set of (propositional) formulae. "
)

RULE = """Some syntactical rules are to be followed:\n
1. Each line consists of only one piece of fact, possibly there could be explanation of the formula after a hashtag #.\n
2. Predicates are named in terms of properties such as location, color, size etc. Connect words with underline. Use lowercases only.\n
3. Objects and quantities are given as arguments of the predicates (use irrel if the object is uncountable or the number is irrelevant). Connect words with underline and always use singular form\n
4. Logical connectives such as AND, OR, or NOT might be used.
5. Description is given in form 'TEXT: (Description of the image)' and output should be in form 'FORMULA: (a set of logical formulae)'.
"""

lang_rules_dict = {
    'juice_bottle': """
6. Use only the following predicates:
color($object$, $color$) # $object$ has a certain $color$
fruit_label($fruit$) # if a label with an icon of $fruit$ is mentioned
volume($description$) # the volume satisfies the $description$
sticker_at($description$, $location$) # a sticker satisfying the $description$ is at the $location$
image_on_label($location$) # the $location$ of the image on the label
juice_fruit_match(irrel) # the color of the juice and the label of fruit match with each other
count($object$, $number$) # the $number$ of the $object$
is_symmetrical(irrel) # The bottle with stickers is symmetrical
""",
########################################################
    'breakfast_box':"""
6. Use only the following predicates:
left($object$, $number$) # the $number$ of the $object$ on the left
right($object$, $number$) # the $number$ of the $object$ on the right
""",
########################################################
    'splicing_connectors':"""
6. Use only the following predicates:
count($object$, $number$) # The $number$ of the $object$
is_broken($object$) # The $object$ is borken somewhere
same_size($object$) # all $object$ in the image appear to be the same size
patch_count($object$, $number$) # the $number$ of the $object$ in a patch description,  
position($object$, $location$) # The $object$ is positioned at $location$
""",
########################################################

}

two_shot_dict = {
    'juice_bottle':  """Some examples are as follows: 

TEXT: - The color of the juice is yellow or wine.\n- The fruit depicted on the label is a pair of cherries.\n- The juice is filled to around half of the neck.\n- There are two stickers.\n    - The top sticker is correct (square sticker with fruit, fruit matches with juice, fruit is located in the middle of the label).\n    - The bottom sticker is correct (100% juice, located at the bottom of the bottle, horizontally centered).\n- The bottle with stickers is symmetrical.

FORMULA: 
color(juice, yellow) OR color(juice, wine) # The color of the juice is yellow or wine.
fruit_label(pair_of_cherries) # The fruit depicted on the label is a pair of cherries.
volume(around_half_neck) #  The juice is filled to around half of the neck.
count(sticker,2) # There are two stickers.
sticker_at(100%_juice, bottom) # The bottom sticker is correct (100% juice, located at the bottom of the bottle
sticker_at(fruit, top) # The top sticker is correct (square sticker with fruit, fruit matches with juice,
juice_fruit_match(irrel)# fruit matches with juice,
image_on_label(middle) # fruit is located in the middle of the label
is_symmetrical(irrel) # The bottle with stickers is symmetrical

TEXT: - The color of the juice is white.\n- The fruit could be banana.\n- The juice is filled to \"full\" in the bottle (more than half of the neck). Are there two stickers?: No stickers\n- Is the top sticker correct: No (Sticker is not present)\n- Is the bottom sticker correct: No (Sticker is not present)\n- Is the bottle with stickers symmetric?: Not applicable (No stickers present)

FORMULA:
color(juice, white) # The color of the juice is white.
fruit_label(banana) # The fruit could be banana.
volume(full) # The juice is filled to \"full\" in the bottle (more than half of the neck).
count(sticker,0) # Are there two stickers?: No stickers
NOT sticker_at(irrel,top) # Is the top sticker correct: No (Sticker is not present)
NOT sticker_at(irrel, bottom) # Is the bottom sticker correct: No (Sticker is not present)
NOT is_symmetrical(irrel) # Is the bottle with stickers symmetric?: Not applicable (No stickers present)


Now output the formulae for another description:
TEXT:  """,

########################################################
    "breakfast_box":"""Some examples are as follows:
TEXT: 
On the left side of the image, there is one apple and two mandarins (can also be oranges or tangerines, clementines). On the right side of the image, there is granola, banana chips (or dried banana chip), and almonds.

FORMULA: 
left(apple,1) # On the left side of the image, there is one apple
left(mandarin,2) OR left(orange,2) OR left(tangerine,2) OR left(clementine,2) On the left side of the image, there are two mandarins (can also be oranges or tangerines, clementines)
right(granola,irrel) # On the right side of the image, there is granola
right(banana_chip,irrel) OR right(dried_banana_chip,irrel) # On the right side of the image, there are banana chips (or dried banana chip), and almonds.
right(amond,irrel) # On the right side of the image, there are almonds.

TEXT:The image shows a food tray with two compartments. 
- On the left side of the image, there are three pieces of fruit: one peach and two mandarins.
- On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.

FORMULA: 
left(peach,1) # On the left side of the image, there is one peach 
left(mandarin,2) # On the left side of the image, there are two mandarins.
right(granola,irrel) # On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.
right(dried_banana,irrel) # On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.
right(almond,irrel) # On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.

Now output the formulae for another description:

TEXT: """,

########################################################

    'splicing_connectors': """

TEXT: 
Here are the answers based on the image provided:
- There are two separate connector blocks.
- There is one cable.
- The cable is not broken.
- Both connector blocks appear to be the same size. patch descriptions: 'The image contains only one block of connectors. The vertical position of the cable is at the top of the connectors.' patch descriptions: 'The image contains only one block of connectors. The vertical position of the cable is at the top of the connectors.'",

FORMULA: 
count(connector_block, 2) # There are two separate connector blocks.
count(cable, 1) # There is one cable.
NOT is_broken(cable) # The cable is not broken.
patch_count(connector_block, 1) # patch descriptions: 'The image contains only one block of connectors.
same_size(connector_block) # Both connector blocks appear to be the same size.
position(cabel, top) # The vertical position of the cable is at the top of the connectors.
patch_count(connector_block, 1) # The vertical position of the cable is at the top of the connectors.
position(cabel, top) # The vertical position of the cable is at the top of the connectors.


TEXT:
The image contains separate connector blocks. 
1. How many connectors are there? 
   - There are two connector blocks in the image.
2. How many cables are there?
   - There is one cable in the image.
3. Is the cable broken or not?
   - The cable appears to be broken in the middle.
4. Do the connectors have the same size?
   - Yes, the connectors appear to be the same size. 
patch descriptions: The image contains only one block of connectors. The cable is positioned at the bottom of the connectors. patch descriptions: 'The image contains only one block of connectors. The vertical position of the cable is at the bottom of the connectors.'

FORMULA: 
count(connector_block, 2) # There are two connector blocks in the image.
count(cable, 1) # There is one cable in the image.
is_broken(cable) # The cable appears to be broken in the middle.
same_size(connector_block) # the connectors appear to be the same size
patch_count(connector_block, 1) # patch descriptions: The image contains only one block of connectors.
position(cabel, bottom) # The cable is positioned at the bottom of the connectors.
patch_count(connector_block, 1) # patch descriptions: 'The image contains only one block of connectors.
position(cabel, top) # The vertical position of the cable is at the top of the connectors.

Now output the formulae for another description:

TEXT: """,

}

# logical_spec_juice = {
#     'norm_spec': """volume(almost_full)& -volume(completely_full)&-volume(three_quarters_full).
# sticker_at(100percent_juice, bottom).
# exists x sticker_at(x,top).
# image_on_label(middle).
# color_match(irrel).
# count(sticker,2).
# """,
#     'norm_pred': ['count', 'image_on_label', 'sticker_at', 'volume','color_match'],
#     'norm_const': ['fruit', 'bottom', 'top', '100percent_juice', 'almost_full', 'three_quarters_full', 'completely_full', 'middle', '2', 'irrel','sticker','0'], # irrel and 0 must be in the list of normal constants
#     'predicate_feat' :{
#         'count': 'bfc',
#         'image_on_label': 'uf',
#         'sticker_at': 'bf',
#         'volume': 'uf',
#         'color_match': 'uf'
#     },
#     'default_values':[['count', 'sticker','0'],['color_match','0']]
# }

logical_spec_juice = {
    'norm_spec': """volume(around_half_neck)& -volume(full)&-volume(lower_than_half_neck)&-volume(empty)&-volume(more_than_half_neck).
sticker_at(100percent_juice, bottom).
exists x sticker_at(x,top).
image_on_label(middle).
juice_fruit_match(irrel).
count(sticker,2).
is_symmetrical(irrel).
""",
    'norm_pred': ['count', 'image_on_label', 'sticker_at', 'volume','juice_fruit_match','is_symmetrical'],
    'norm_const': ['fruit', 'bottom', 'top', '100percent_juice', 'around_half_neck', 'full', 'more_than_half_neck','lower_than_half_neck', 'largely_empty', 'middle', '2', 'irrel','sticker','0'], # irrel and 0 must be in the list of normal constants
    'predicate_feat' :{
        'count': 'bfc',
        'image_on_label': 'uf',
        'sticker_at': 'bf',
        'volume': 'uf',
        'juice_fruit_match': 'uf',
        'is_symmetrical':'u'
    },
    'default_values':[['count', 'sticker','0'],['juice_fruit_match','0']]
}

logical_spec_splicing = {
    'norm_spec':"""count(connector_block,2).
count(cable,1).
-is_broken(cable).
patch_count(connector_block,1).
same_size(connector_block).
""",
    'norm_pred': ['count', 'is_broken', 'patch_count', 'same_size','position'],
    'norm_const': ['connector_slot','connector_block', 'cable','irrel','1', '2', '0'], # irrel and 0 must be in the list of normal constants
    'predicate_feat' :{
        'count': 'bf',
        'is_broken': 'u',
        'patch_count': 'bf',
        'same_size': 'u',
        'position': 'bf'
    },
    'default_values': [['count', 'connector_block', '0'], 
                       ['count', 'cable', '0'],
                       ['patch_count', 'connector_block', '0']]
}

logical_spec_breakfast = {
    'norm_spec':"""left(orange,2).
(left(apple,1)&left(nectarine,0))|(left(apple,0)&left(nectarine,1)).
right(cereal,irrel).
right(banana_chip, irrel).
right(nut, irrel).
""",
    'norm_pred': ['left', 'right'],
    'norm_const': ['orange', 'apple','nectarine', 'cereal', 'banana_chip','nut','irrel','1', '2', '0'], # irrel and 0 must be in the list of normal constants
    'predicate_feat' :{
        'left': 'bfc',
        'right': 'bfc',
    },
    'default_values': [['left', 'orange', '0'], ['left', 'nectarine', '0'], ['left', 'apple', '0'],['right', 'cereal', '0'], ['right', 'banana_chip', '0'],['right', 'nut', '0'],]
}

logical_spec_dict = {
    'juice_bottle': logical_spec_juice,
    'splicing_connectors': logical_spec_splicing,
    'breakfast_box': logical_spec_breakfast
}


natural_normal_spec = {
    'breakfast_box': """
On the left there should be either an apple and no nectarine, or a necdtarine but no apples.
On the right there is cereal, and the number is irrelevant.
On the right there is some banana_chips, where the nubmer is irrelevant.
On the right there is some nuts, where the number is irrelevant.
""",  
}
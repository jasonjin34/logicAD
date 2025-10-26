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
color_match(irrel) # the color of the juice and the label of fruit match with each other
count($object$, $number$) # the $number$ of the $object$
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

TEXT: The juice in the bottle appears to be a reddish or wine color, which is appropriate for cherry juice. The fruit depicted on the label is a pair of cherries. The bottle appears to be mostly full, with a small air gap at the top. Yes, there are two stickers on the bottle: one at the bottom with the text \"100% Juice\" and another at the top with the illustration of the cherries. The cherries are located approximately in the middle of the top label.

FORMULA: 
color(juice, reddish) OR color(juice, wine) # The juice in the bottle appears to be a reddish or wine color.
fruit_label(pair_of_cherries) # The fruit depicted on the label is a pair of cherries.
color_match(irrel)# The juice in the bottle appears to be a reddish or wine color, which is appropriate for cherry juice.
volume(mostly_full) # The bottle appears to be mostly full
count(sticker,2) # there are two stickers on the bottle
sticker_at(100%_juice, bottom) # there are two stickers on the bottle: one at the bottom with the text \"100% Juice\"
sticker_at(cherry, top) # another at the top with the illustration of the cherries
image_on_label(middle) # The cherries are located approximately in the middle of the top label.

TEXT:
Based on the image:
1. **Color of the Juice**: The juice is a reddish-brown color.
2. **Possible Fruit**: Given the reddish-brown color, it could be juice from a fruit like pomegranate, cherry, or a blend of fruits with similar colors.
3. **Amount of Juice in the Bottle**: The bottle appears to be filled almost up to the neck.
4. **Labels/Stickering**: The bottle does not have any visible stickers or labels that indicate \"100% juice\" or the type of fruit. There is no image of a fruit on the label, nor are there any other identifying features present on the bottle.

The bottle shown is likely intended for multiple types of juice, but any further specifics about the stickers or labels cannot be confirmed visually.

FORMULA:
color(juice, reddish_brown) # The juice is a reddish-brown color.
volume(almost_up_to_neck) # **Amount of Juice in the Bottle**: The bottle appears to be filled almost up to the neck
NOT sticker_at(100%_juice, irrel) # The bottle does not have any visible stickers or labels that indicate \"100% juice\"
NOT image_on_label(irrel) # There is no image of a fruit on the label

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

logical_spec_juice = {
    'norm_spec': """volume(almost_full)& -volume(completely_full)&-volume(three_quarters_full).
sticker_at(100percent_juice, bottom).
exists x sticker_at(x,top).
image_on_label(middle).
color_match(irrel).
count(sticker,2).
""",
    'norm_pred': ['count', 'image_on_label', 'sticker_at', 'volume','color_match'],
    'norm_const': ['fruit', 'bottom', 'top', '100percent_juice', 'almost_full', 'three_quarters_full', 'completely_full', 'middle', '2', 'irrel','sticker','0'], # irrel and 0 must be in the list of normal constants
    'predicate_feat' :{
        'count': 'bfc',
        'image_on_label': 'uf',
        'sticker_at': 'bf',
        'volume': 'uf',
        'color_match': 'uf'
    },
    'default_values':[['count', 'sticker','0'],['color_match','0']]
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
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import nltk
from nltk import PCFG
import numpy as np
from scipy.stats import zipf
from scipy.special import softmax


# # Vocabulary

# In[ ]:


# 100 nouns, picked from the MacArthur Communicative Development Inventory and the BNC top frequent nouns
# BNC freq rank: http://ucrel.lancs.ac.uk/bncfreq/flists.html
animate_nouns = [
    'girl', 'boy', 'cat', 'dog', 'baby', 'child', 'teacher', 'frog', 'chicken', 'mouse',
    'lion', 'monkey', 'bear', 'giraffe', 'horse', 'bird', 'duck', 'bunny', 'butterfly', 'penguin',
    'student', 'professor', 'monster', 'hero', 'sailor', 'lawyer', 'customer', 'scientist', 'princess', 'president',
    'cow', 'crocodile', 'goose', 'hen', 'deer', 'donkey', 'bee', 'fly', 'kitty', 'tiger',
    'wolf', 'zebra', 'mother', 'father', 'patient', 'manager', 'director', 'king', 'queen', 'kid',
    'fish', 'moose',  'pig', 'pony', 'puppy', 'sheep', 'squirrel', 'lamb', 'turkey', 'turtle', 
    'doctor', 'pupil', 'prince', 'driver', 'consumer', 'writer', 'farmer', 'friend', 'judge', 'visitor',
    'guest', 'servant', 'chief', 'citizen', 'champion', 'prisoner', 'captain', 'soldier', 'passenger', 'tenant',
    'politician', 'resident', 'buyer', 'spokesman', 'governor', 'guard', 'creature', 'coach', 'producer', 'researcher',
    'guy', 'dealer', 'duke', 'tourist', 'landlord', 'human', 'host', 'priest', 'journalist', 'poet'
]
assert len(set(animate_nouns)) == 100

inanimate_nouns = [
    'cake', 'donut', 'cookie', 'box', 'rose', 'drink', 'raisin', 'melon', 'sandwich', 'strawberry', 
    'ball', 'balloon', 'bat', 'block', 'book', 'crayon', 'chalk', 'doll', 'game', 'glue',
    'lollipop', 'hamburger', 'banana', 'biscuit', 'muffin', 'pancake', 'pizza', 'potato', 'pretzel', 'pumpkin',
    'sweetcorn', 'yogurt', 'pickle', 'jigsaw', 'pen', 'pencil', 'present', 'toy', 'cracker', 'brush',
    'radio', 'cloud', 'mandarin', 'hat', 'basket', 'plant', 'flower', 'chair', 'spoon', 'pillow',
    'gumball', 'scarf', 'shoe', 'jacket', 'hammer', 'bucket', 'knife', 'cup', 'plate', 'towel',
    'bottle', 'bowl', 'can', 'clock', 'jar', 'penny', 'purse', 'soap', 'toothbrush', 'watch',
    'newspaper', 'fig', 'bag', 'wine', 'key', 'weapon', 'brain', 'tool', 'crown', 'ring',
    'leaf', 'fruit', 'mirror', 'beer', 'shirt', 'guitar', 'chemical', 'seed', 'shell', 'brick',
    'bell', 'coin', 'button', 'needle', 'molecule', 'crystal', 'flag', 'nail', 'bean', 'liver'
]

assert len(set(inanimate_nouns)) == 100

# 100 names, picked from https://www.ssa.gov/OACT/babynames/
proper_nouns = [
    'Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'William', 'Isabella', 'James', 'Sophia', 'Oliver', 
    'Charlotte', 'Benjamin', 'Mia', 'Elijah', 'Amelia', 'Lucas', 'Harper', 'Mason', 'Evelyn', 'Logan',
    'Abigail', 'Alexander', 'Emily', 'Ethan', 'Elizabeth', 'Jacob', 'Mila', 'Michael', 'Ella', 'Daniel',
    'Avery', 'Henry', 'Sofia', 'Jackson', 'Camila', 'Sebastian', 'Aria', 'Aiden', 'Scarlett', 'Matthew',
    'Victoria', 'Samuel', 'Madison', 'David', 'Luna', 'Joseph', 'Grace', 'Carter', 'Chloe', 'Owen',
    'Penelope', 'Wyatt', 'Layla', 'John', 'Riley', 'Jack', 'Zoey', 'Luke', 'Nora', 'Jayden',
    'Lily', 'Dylan', 'Eleanor', 'Grayson', 'Hannah', 'Levi', 'Lillian', 'Isaac', 'Addison', 'Gabriel',
    'Aubrey', 'Julian', 'Ellie', 'Mateo', 'Stella', 'Anthony', 'Natalie', 'Jaxon', 'Zoe', 'Lincoln',
    'Leah', 'Joshua', 'Hazel', 'Christopher', 'Violet', 'Andrew', 'Aurora', 'Theodore', 'Savannah', 'Caleb',
    'Audrey', 'Ryan', 'Brooklyn', 'Asher', 'Bella', 'Nathan', 'Claire', 'Thomas', 'Skylar', 'Leo'
]

assert len(set(proper_nouns)) == 100

# P + N: N from BNC + COCA

# 100 nouns that can appear with "on" 
on_nouns = [
    'table', 'stage', 'bed', 'chair', 'stool', 'road', 'tree', 'box', 'surface', 'seat',
    'speaker', 'computer', 'rock', 'boat', 'cabinet', 'TV', 'plate', 'desk', 'bowl', 'bench',
    'shelf', 'cloth', 'piano', 'bible', 'leaflet', 'sheet', 'cupboard', 'truck', 'tray', 'notebook',
    'blanket', 'deck', 'coffin', 'log', 'ladder', 'barrel', 'rug', 'canvas', 'tiger', 'towel',
    'throne', 'booklet', 'sock', 'corpse', 'sofa', 'keyboard', 'book', 'pillow', 'pad', 'train',
    'couch', 'bike', 'pedestal', 'platter', 'paper', 'rack', 'board', 'panel', 'tripod', 'branch',
    'machine', 'floor', 'napkin', 'cookie', 'block', 'cot', 'device', 'yacht', 'dog', 'mattress',
    'ball', 'stand', 'stack', 'windowsill', 'counter', 'cushion', 'hanger', 'trampoline', 'gravel', 'cake',
    'carpet', 'plaque', 'boulder', 'leaf', 'mound', 'bun', 'dish', 'cat', 'podium', 'tabletop',
    'beach', 'bag', 'glacier', 'brick', 'crack', 'vessel', 'futon', 'turntable', 'rag', 'chessboard'
]

# 100 nouns that can appear with "in"
in_nouns = [
    'house', 'room', 'car', 'garden', 'box', 'cup', 'glass', 'bag', 'vehicle', 'hole',
    'cabinet', 'bottle', 'shoe', 'storage', 'cot', 'vessel', 'pot', 'pit', 'tin', 'can',
    'cupboard', 'envelope', 'nest', 'bush', 'coffin', 'drawer', 'container', 'basin', 'tent', 'soup',
    'well', 'barrel', 'bucket', 'cage', 'sink', 'cylinder', 'parcel', 'cart', 'sack', 'trunk',
    'wardrobe', 'basket', 'bin', 'fridge', 'mug', 'jar', 'corner', 'pool', 'blender', 'closet',
    'pile', 'van', 'trailer', 'saucepan', 'truck', 'taxi', 'haystack', 'dumpster', 'puddle', 'bathtub',
    'pod', 'tub', 'trap', 'bun', 'microwave', 'bookstore', 'package', 'cafe', 'train', 'castle',
    'bunker', 'vase', 'backpack', 'tube', 'hammock', 'stadium', 'backyard', 'swamp', 'monastery', 'refrigerator',
    'palace', 'cubicle', 'crib', 'condo', 'tower', 'crate', 'dungeon', 'teapot', 'tomb', 'casket',
    'jeep', 'shoebox', 'wagon', 'bakery', 'fishbowl', 'kennel', 'china', 'spaceship', 'penthouse', 'pyramid'
] 

# 100 nouns that can appear with "beside"
beside_nouns = [
    'table', 'stage', 'bed', 'chair', 'book', 'road', 'tree', 'machine', 'house', 'seat',
    'speaker', 'computer', 'rock', 'car', 'box', 'cup', 'glass', 'bag', 'flower', 'boat',
    'vehicle', 'key', 'painting', 'cabinet', 'TV', 'bottle', 'cat', 'desk', 'shoe', 'mirror',
    'clock', 'bench', 'bike', 'lamp', 'lion', 'piano', 'crystal', 'toy', 'duck', 'sword',
    'sculpture', 'rod', 'truck', 'basket', 'bear', 'nest', 'sphere', 'bush', 'surgeon', 'poster',
    'throne', 'giant', 'trophy', 'hedge', 'log', 'tent', 'ladder', 'helicopter', 'barrel', 'yacht',
    'statue', 'bucket', 'skull', 'beast', 'lemon', 'whale', 'cage', 'gardner', 'fox', 'sink',
    'trainee', 'dragon', 'cylinder', 'monk', 'bat', 'headmaster', 'philosopher', 'foreigner', 'worm', 'chemist',
    'corpse', 'wolf', 'torch', 'sailor', 'valve', 'hammer', 'doll', 'genius', 'baron', 'murderer',
    'bicycle', 'keyboard', 'stool', 'pepper', 'warrior', 'pillar', 'monkey', 'cassette', 'broker', 'bin'
    
]

assert len(set(on_nouns)) == len(set(in_nouns)) == len(set(beside_nouns)) == 100
noun_list = animate_nouns + inanimate_nouns + proper_nouns + on_nouns + in_nouns + beside_nouns
# print(len(set(noun_list)))

# Levin, '1.2.1 Unspecified Object Alternation'
# And some intuition-based selection. 
V_trans_omissible = [
  'ate', 'painted', 'drew', 'cleaned', 'cooked', 
  'dusted', 'hunted', 'nursed', 'sketched', 'juggled',
  'called', 'heard', 'packed', 'saw', 'noticed',
  'studied', 'examined', 'observed', 'knew', 'investigated'
]
V_trans_omissible_pp = [
  'eaten', 'painted', 'drawn', 'cleaned', 'cooked',
  'dusted', 'hunted', 'nursed', 'sketched', 'juggled',
  'called', 'heard', 'packed', 'seen', 'noticed',
  'studied', 'examined', 'observed', 'known', 'investigated'
]

assert len(set(V_trans_omissible)) == len(set(V_trans_omissible_pp)) == 20 

# Levin class 30. Verbs of Perception, 31.2 Admire Verbs, VerbNet poke-19, throw-17.1.1
V_trans_not_omissible = [
  'liked', 'helped', 'found', 'loved', 'poked',
  'admired', 'adored', 'appreciated', 'missed', 'respected',
  'threw', 'tolerated', 'valued', 'worshipped', 'discovered', 
  'held', 'stabbed', 'touched', 'pierced', 'tossed'
]
V_trans_not_omissible_pp = [
  'liked', 'helped', 'found', 'loved', 'poked', 
  'admired', 'adored', 'appreciated', 'missed', 'respected', 
  'thrown', 'tolerated', 'valued', 'worshipped', 'discovered', 
  'held', 'stabbed', 'touched', 'pierced', 'tossed'
]

assert set(V_trans_omissible).isdisjoint(set(V_trans_not_omissible))
assert set(V_trans_omissible_pp).isdisjoint(set(V_trans_not_omissible_pp))

assert len(set(V_trans_not_omissible)) == len(set(V_trans_not_omissible_pp)) == 20 

# Levin 29.4 Declare verbs, Levin 30. Verbs of Perception, VerbNet admire-31.2, VerbNet wish-62
V_cp_taking = [
  'liked', 'hoped', 'said', 'noticed', 'believed',
  'confessed', 'declared', 'proved', 'thought', 'admired',
  'appreciated', 'respected', 'supported', 'tolerated', 'valued',
  'wished', 'dreamed', 'expected', 'imagined', 'meant'
]

assert len(set(V_cp_taking)) == 20
 
# VerbNet want-32.1, VerbNet try-61, VerbNet wish-62, VerbNet long-32.2, VerbNet admire-31.2-1
V_inf_taking = [
  'wanted', 'preferred', 'needed', 'intended', 'tried',
  'attempted', 'planned', 'expected', 'hoped', 'wished', 
  'craved', 'liked', 'hated', 'loved', 'enjoyed',
  'dreamed', 'meant', 'longed', 'yearned', 'itched'
]
assert len(set(V_inf_taking)) == 20

# 1.1.2.1 Causative-Inchoative Alternation
V_unacc = [
  'rolled', 'froze', 'burned', 'shortened', 'floated', 
  'grew', 'slid', 'broke', 'crumpled', 'split', 
  'changed', 'snapped', 'disintegrated', 'collapsed', 'decomposed',
  'doubled', 'improved', 'inflated', 'enlarged', 'reddened', 
]
V_unacc_pp = [
  'rolled', 'frozen', 'burned', 'shortened', 'floated',
  'grown', 'slid', 'broken', 'crumpled', 'split',
  'changed', 'snapped', 'disintegrated', 'collapsed', 'decomposed', 
  'doubled', 'improved', 'inflated', 'enlarged', 'reddened'
]
assert len(set(V_unacc)) == len(set(V_unacc_pp)) == 20

V_unerg = [
  'slept', 'smiled', 'laughed', 'sneezed', 'cried', 
  'talked', 'danced', 'jogged', 'walked', 'ran', 
  'napped', 'snoozed', 'screamed', 'stuttered', 'frowned', 
  'giggled', 'scoffed', 'snored', 'smirked', 'gasped'
]
assert len(set(V_unerg)) == 20

# 10 DO omissible transitives, 10 unergatives
V_inf = [
  'walk', 'run', 'sleep', 'sneeze', 'nap',
  'eat', 'read', 'cook', 'hunt', 'paint',
  'talk', 'dance', 'giggle', 'jog', 'smirk',
  'call', 'sketch', 'dust', 'clean', 'investigate'
]
assert len(set(V_inf)) == 20

V_dat = [
  'gave', 'lended', 'sold', 'offered', 'fed',
  'passed', 'sent', 'rented', 'served', 'awarded', 
  'brought', 'handed', 'forwarded', 'promised', 'mailed',
  'loaned', 'posted', 'returned', 'slipped', 'wired'
]
V_dat_pp = [
  'given', 'lended', 'sold', 'offered', 'fed', 
  'passed', 'sent', 'rented', 'served', 'awarded',
  'brought', 'handed', 'forwarded', 'promised', 'mailed', 
  'loaned', 'posted', 'returned', 'slipped', 'wired'
]

assert len(set(V_dat)) == len(set(V_dat_pp)) == 20

# print(len(set(V_trans_omissible + V_trans_not_omissible + V_cp_taking + V_unacc + V_unerg + V_dat)))

verbs_lemmas = { 
  'ate':'eat', 'painted':'paint', 'drew':'draw', 'cleaned':'clean',
  'cooked':'cook', 'dusted':'dust', 'hunted':'hunt', 'nursed':'nurse',
  'sketched':'sketch', 'washed':'wash', 'juggled':'juggle', 'called':'call',
  'eaten':'eat', 'drawn':'draw', 'baked':'bake', 'liked':'like', 'knew':'know', 
  'helped':'help', 'saw':'see', 'found':'find', 'heard':'hear', 'noticed':'notice',
  'loved':'love', 'admired':'admire', 'adored':'adore', 'appreciated':'appreciate',
  'missed':'miss', 'respected':'respect', 'tolerated':'tolerate', 'valued':'value', 
  'worshipped':'worship', 'observed':'observe', 'discovered':'discover', 'held':'hold',
  'stabbed':'stab', 'touched':'touch', 'pierced':'pierce', 'poked':'poke',
  'known':'know', 'seen':'see', 'hit':'hit', 'hoped':'hope', 'said':'say',
  'believed':'believe', 'confessed':'confess', 'declared':'declare', 'proved':'prove',
  'thought':'think', 'supported':'support', 'wished':'wish', 'dreamed':'dream', 
  'expected':'expect', 'imagined':'imagine', 'envied':'envy', 'wanted':'want', 
  'preferred':'prefer', 'needed':'need', 'intended':'intend', 'tried':'try',
  'attempted':'attempt', 'planned':'plan','craved':'crave','hated':'hate','loved':'love', 
  'enjoyed':'enjoy', 'rolled':'roll', 'froze':'freeze', 'burned':'burn', 'shortened':'shorten',
  'floated':'float', 'grew':'grow', 'slid':'slide', 'broke':'break', 'crumpled':'crumple',
  'split':'split', 'changed':'change', 'snapped':'snap', 'tore':'tear', 'collapsed':'collapse',
  'decomposed':'decompose', 'doubled':'double', 'improved':'improve', 'inflated':'inflate',
  'enlarged':'enlarge', 'reddened':'redden', 'popped':'pop', 'disintegrated':'disintegrate',
  'expanded':'expand', 'cooled':'cool', 'soaked':'soak', 'frozen':'freeze', 'grown':'grow',
  'broken':'break', 'torn':'tear', 'slept':'sleep', 'smiled':'smile', 'laughed':'laugh',
  'sneezed':'sneeze', 'cried':'cry', 'talked':'talk', 'danced':'dance', 'jogged':'jog',
  'walked':'walk', 'ran':'run', 'napped':'nap', 'snoozed':'snooze', 'screamed':'scream',
  'stuttered':'stutter', 'frowned':'frown', 'giggled':'giggle', 'scoffed':'scoff',
  'snored':'snore', 'snorted':'snort', 'smirked':'smirk', 'gasped':'gasp',
  'gave':'give', 'lended':'lend', 'sold':'sell', 'offered':'offer', 'fed':'feed', 
  'passed':'pass', 'rented':'rent', 'served':'serve','awarded':'award', 'promised':'promise',
  'brought':'bring', 'sent':'send', 'handed':'hand', 'forwarded':'forward', 'mailed':'mail',
  'posted':'post','given':'give', 'shipped':'ship', 'packed':'pack', 'studied':'study', 
  'examined':'examine', 'investigated':'investigate', 'thrown':'throw', 'threw':'throw',
  'tossed':'toss', 'meant':'mean', 'longed':'long', 'yearned':'yearn', 'itched':'itch',
  'loaned':'loan', 'returned':'return', 'slipped':'slip', 'wired':'wire', 'crawled':'crawl',
  'shattered':'shatter', 'bought':'buy', 'squeezed':'squeeze', 'teleported':'teleport',
  'melted':'melt', 'blessed':'bless'
}

pos_d = {
    'a': 'DET',
    'the': 'DET',
    'to': 'ADP',
    'on': 'ADP',
    'in': 'ADP',
    'beside': 'ADP',
    'that': 'SCONJ',
    'was': 'AUX',
    'by': 'ADP'
}


# # Vocab items for gen set

# In[ ]:


only_seen_as_subject = 'hedgehog'
only_seen_as_noun_prim = 'shark'
only_seen_as_object = 'cockroach'
only_seen_as_subject_proper_noun = 'Lina'
only_seen_as_proper_noun_prim = 'Paula'
only_seen_as_object_proper_noun =  'Charlie'
only_seen_as_transitive_obj_omissible = 'baked'
only_seen_as_unaccuative = 'shattered'
only_seen_as_verb_prim = 'crawl'
only_seen_as_transitive_subject_animate = 'cobra'
only_seen_as_unaccusative_subject_animate = 'hippo'
only_seen_as_active = 'blessed'
only_seen_as_passive = 'squeezed'
only_seen_as_double_object = 'teleported'
only_seen_as_pp = 'shipped'


target_item_nouns = [only_seen_as_subject, only_seen_as_noun_prim, only_seen_as_object,
                    only_seen_as_transitive_subject_animate, only_seen_as_unaccusative_subject_animate]

target_item_props = [only_seen_as_subject_proper_noun, only_seen_as_proper_noun_prim,
                    only_seen_as_object_proper_noun]

pos_d.update({n: 'PROPN' for n in proper_nouns + target_item_props})
pos_d.update({n: 'NOUN' for n in noun_list + target_item_nouns})


# # Assign Zipfian distribution to vocab

# In[ ]:


def normalize(probs):
    leftover_prob = 1-sum(probs)
    probs = probs + leftover_prob/len(probs)
    return probs

# fig, ax = plt.subplots(1, 1)
a = 1.4
k = np.array(list(range(1,101)))
# print(sum(zipf.pmf(k, a)))
# print(len(zipf.pmf(k, a)))
# 
# print(zipf.pmf(np.array(range(1,len(animate_nouns)+1)), a))
# print(sum(zipf.pmf(np.array(range(1,len(animate_nouns)+1)), a)))
 
animate_nouns_prob = zipf.pmf(np.array(range(1,len(animate_nouns)+1)), a)
inanimate_nouns_prob = zipf.pmf(np.array(range(1,len(inanimate_nouns)+1)), a)
proper_nouns_prob = zipf.pmf(np.array(range(1,len(proper_nouns)+1)), a)
in_nouns_prob = zipf.pmf(np.array(range(1,len(in_nouns)+1)), a)
on_nouns_prob = zipf.pmf(np.array(range(1,len(on_nouns)+1)), a)
beside_nouns_prob = zipf.pmf(np.array(range(1,len(beside_nouns)+1)), a)

V_trans_omissible_prob = zipf.pmf(np.array(range(1,len(V_trans_omissible)+1)), a)
V_trans_omissible_pp_prob = zipf.pmf(np.array(range(1,len(V_trans_omissible_pp)+1)), a)
V_trans_not_omissible_prob = zipf.pmf(np.array(range(1,len(V_trans_not_omissible)+1)), a)
V_trans_not_omissible_pp_prob = zipf.pmf(np.array(range(1,len(V_trans_not_omissible_pp)+1)), a)
V_cp_taking_prob = zipf.pmf(np.array(range(1,len(V_cp_taking)+1)), a)
V_inf_taking_prob = zipf.pmf(np.array(range(1,len(V_inf_taking)+1)), a)
V_unacc_prob = zipf.pmf(np.array(range(1,len(V_unacc)+1)), a)
V_unacc_pp_prob = zipf.pmf(np.array(range(1,len(V_unacc_pp)+1)), a)
V_unerg_prob = zipf.pmf(np.array(range(1,len(V_unerg)+1)), a)
V_inf_prob = zipf.pmf(np.array(range(1,len(V_inf)+1)), a)
V_dat_prob = zipf.pmf(np.array(range(1,len(V_dat)+1)), a)
V_dat_pp_prob = zipf.pmf(np.array(range(1,len(V_dat_pp)+1)), a)

# make probs sum to one
animate_nouns_prob = normalize(animate_nouns_prob)
inanimate_nouns_prob = normalize(inanimate_nouns_prob)
proper_nouns_prob = normalize(proper_nouns_prob)
in_nouns_prob = normalize(in_nouns_prob)
on_nouns_prob = normalize(on_nouns_prob)
beside_nouns_prob = normalize(beside_nouns_prob)
V_trans_omissible_prob = normalize(V_trans_omissible_prob)
V_trans_omissible_pp_prob = normalize(V_trans_omissible_pp_prob)
V_trans_not_omissible_prob = normalize(V_trans_not_omissible_prob)
V_trans_not_omissible_pp_prob = normalize(V_trans_not_omissible_pp_prob)
V_cp_taking_prob = normalize(V_cp_taking_prob)
V_inf_taking_prob = normalize(V_inf_taking_prob)
V_unacc_prob = normalize(V_unacc_prob)
V_unacc_pp_prob = normalize(V_unacc_pp_prob)
V_unerg_prob = normalize(V_unerg_prob)
V_inf_prob = normalize(V_inf_prob)
V_dat_prob = normalize(V_dat_prob)
V_dat_pp_prob = normalize(V_dat_pp_prob)

# ax.plot(k, proper_nouns_prob, 'bo', ms=8, label='zipf pmf')

animate_nouns_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(animate_nouns, animate_nouns_prob)])
inanimate_nouns_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(inanimate_nouns, inanimate_nouns_prob)])
proper_nouns_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(proper_nouns, proper_nouns_prob)])
in_nouns_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(in_nouns, in_nouns_prob)])
on_nouns_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(on_nouns, on_nouns_prob)])
beside_nouns_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(beside_nouns, beside_nouns_prob)])

V_trans_omissible_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_trans_omissible, V_trans_omissible_prob)])
V_trans_omissible_pp_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_trans_omissible_pp, V_trans_omissible_pp_prob)])
V_trans_not_omissible_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_trans_not_omissible, V_trans_not_omissible_prob)])
V_trans_not_omissible_pp_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_trans_not_omissible_pp, V_trans_not_omissible_pp_prob)])
V_cp_taking_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_cp_taking, V_cp_taking_prob)])
V_inf_taking_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_inf_taking, V_inf_taking_prob)])
V_unacc_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_unacc, V_unacc_prob)])
V_unacc_pp_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_unacc_pp, V_unacc_pp_prob)])
V_unerg_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_unerg, V_unerg_prob)])
V_inf_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_inf,V_inf_prob)])
V_dat_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_dat, V_dat_prob)])
V_dat_pp_str = " | ".join(["'{}' [{}]".format(n,p) for n, p in zip(V_dat_pp, V_dat_pp_prob)])


# # Main grammar: generates sentences in train/dev/test.

# In[ ]:


main_grammar_str = """

S -> NP_animate_nsubj VP_external [0.49] | VP_internal [0.06] \
   | NP_inanimate_nsubjpass VP_passive [0.375] | NP_animate_nsubjpass VP_passive_dat [0.075]
VP_external -> V_unerg [0.10525] | V_unacc NP_dobj [0.10525] \
             | V_trans_omissible [0.10525] | V_trans_omissible NP_dobj [0.10525] \
             | V_trans_not_omissible NP_dobj [0.10525] | V_inf_taking INF V_inf [0.10525] \
             | V_cp_taking C S [0.158] \
             | V_dat NP_inanimate_dobj PP_iobj [0.10525] | V_dat NP_animate_iobj NP_inanimate_dobj [0.10525]
VP_internal -> NP_unacc_subj V_unacc [1.0]
VP_passive -> AUX V_trans_not_omissible_pp [0.125] | AUX V_trans_not_omissible_pp BY NP_animate_nsubj [0.125] |  \
              AUX V_trans_omissible_pp [0.125] | AUX V_trans_omissible_pp BY NP_animate_nsubj [0.125] |  \
              AUX V_unacc_pp [0.125] | AUX V_unacc_pp BY NP_animate_nsubj [0.125] | \
              AUX V_dat_pp PP_iobj [0.125] | AUX V_dat_pp PP_iobj BY NP_animate_nsubj [0.125]
VP_passive_dat -> AUX V_dat_pp NP_inanimate_dobj [0.5] | AUX V_dat_pp NP_inanimate_dobj BY NP_animate_nsubj [0.5]
NP_dobj -> NP_inanimate_dobj [0.5] | NP_animate_dobj [0.5]
NP_unacc_subj -> NP_inanimate_dobj_noPP [0.5] | NP_animate_dobj_noPP [0.5]
NP_animate_dobj_noPP -> Det N_common_animate_dobj [0.5] | N_prop_dobj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj [0.25] | Det N_common_animate_dobj PP_loc [0.25] \
                 | N_prop_dobj [0.50]
NP_animate_iobj -> Det N_common_animate_iobj [0.5] | N_prop_iobj [0.5]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_animate_nsubjpass -> Det N_common_animate_nsubjpass [0.5] | N_prop_nsubjpass [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [0.5] | Det N_common_inanimate_dobj PP_loc [0.5]
NP_inanimate_dobj_noPP -> Det N_common_inanimate_dobj [1.0]
NP_inanimate_nsubjpass -> Det N_common_inanimate_nsubjpass [1.0]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
AUX -> 'was' [1.0]
BY -> 'by' [1.0]
N_common_animate_dobj -> {animate_nouns_str}
N_common_animate_iobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_animate_nsubjpass -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_common_inanimate_nsubjpass -> {inanimate_nouns_str}
N_prop_dobj -> {proper_nouns_str}
N_prop_iobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
N_prop_nsubjpass -> {proper_nouns_str}
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
V_trans_omissible -> {V_trans_omissible_str}
V_trans_omissible_pp -> {V_trans_omissible_pp_str}
V_trans_not_omissible -> {V_trans_not_omissible_str}
V_trans_not_omissible_pp -> {V_trans_not_omissible_pp_str}
V_cp_taking -> {V_cp_taking_str}
V_inf_taking -> {V_inf_taking_str}
V_unacc -> {V_unacc_str}
V_unacc_pp -> {V_unacc_pp_str}
V_unerg -> {V_unerg_str}
V_inf -> {V_inf_str}
V_dat -> {V_dat_str}
V_dat_pp -> {V_dat_pp_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
INF -> 'to' [1.0]
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,
           V_trans_omissible_str=V_trans_omissible_str, 
           V_trans_omissible_pp_str=V_trans_omissible_pp_str, 
           V_trans_not_omissible_str=V_trans_not_omissible_str, 
           V_trans_not_omissible_pp_str=V_trans_not_omissible_pp_str, 
           V_cp_taking_str=V_cp_taking_str, 
           V_inf_taking_str=V_inf_taking_str, 
           V_unacc_str=V_unacc_str, 
           V_unacc_pp_str=V_unacc_pp_str, 
           V_unerg_str=V_unerg_str, 
           V_inf_str=V_inf_str, 
           V_dat_str=V_dat_str,
           V_dat_pp_str=V_dat_pp_str
          )


# # Subsets of main grammar that only generate target common noun/proper noun as subjects

# In[ ]:


common_noun_subject_grammar_str = """

S -> NP_animate_nsubj_targeted VP_external [0.923] | NP_animate_nsubj VP_CP [0.077]
VP_external -> V_unerg [0.125] | V_unacc NP_dobj [0.125] \
             | V_trans_omissible [0.125] | V_trans_omissible NP_dobj [0.125] \
             | V_trans_not_omissible NP_dobj [0.125] | V_inf_taking INF V_inf [0.125] \
             | V_dat NP_inanimate_dobj PP_iobj [0.125] | V_dat NP_animate_iobj NP_inanimate_dobj [0.125]
VP_CP -> V_cp_taking C S [1.0]
NP_dobj -> NP_inanimate_dobj [0.5] | NP_animate_dobj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj [0.25] | Det N_common_animate_dobj PP_loc [0.25] \
                 | N_prop_dobj [0.50]
NP_animate_iobj -> Det N_common_animate_iobj [0.5] | N_prop_iobj [0.5]
NP_animate_nsubj_targeted -> Det N_common_animate_nsubj_targeted [1.0]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [0.5] | Det N_common_inanimate_dobj PP_loc [0.5]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
AUX -> 'was' [1.0]
BY -> 'by' [1.0]
N_common_animate_dobj -> {animate_nouns_str}
N_common_animate_iobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_animate_nsubj_targeted -> {target_item_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_dobj -> {proper_nouns_str}
N_prop_iobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
V_trans_omissible -> {V_trans_omissible_str}
V_trans_not_omissible -> {V_trans_not_omissible_str}
V_unacc -> {V_unacc_str}
V_cp_taking -> {V_cp_taking_str}
V_inf_taking -> {V_inf_taking_str}
V_unerg -> {V_unerg_str}
V_inf -> {V_inf_str}
V_dat -> {V_dat_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
INF -> 'to' [1.0]
""".format(animate_nouns_str=animate_nouns_str,
           inanimate_nouns_str=inanimate_nouns_str,
           proper_nouns_str=proper_nouns_str,
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,
           V_trans_omissible_str=V_trans_omissible_str,
           V_trans_not_omissible_str=V_trans_not_omissible_str,
           V_cp_taking_str=V_cp_taking_str,
           V_inf_taking_str=V_inf_taking_str,
           V_unerg_str=V_unerg_str,
           V_inf_str=V_inf_str,
           V_dat_str=V_dat_str,
           V_unacc_str=V_unacc_str,
           target_item_str='{}'
          )

proper_noun_subject_grammar_str = """

S -> NP_animate_nsubj_targeted VP_external [0.923] | NP_animate_nsubj VP_CP [0.077]
VP_external -> V_unerg [0.125] | V_unacc NP_dobj [0.125] \
             | V_trans_omissible [0.125] | V_trans_omissible NP_dobj [0.125] \
             | V_trans_not_omissible NP_dobj [0.125] | V_inf_taking INF V_inf [0.125] \
             | V_dat NP_inanimate_dobj PP_iobj [0.125] | V_dat NP_animate_iobj NP_inanimate_dobj [0.125]
VP_CP -> V_cp_taking C S [1.0]
NP_dobj -> NP_inanimate_dobj [0.5] | NP_animate_dobj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj [0.25] | Det N_common_animate_dobj PP_loc [0.25] \
                 | N_prop_dobj [0.50]
NP_animate_iobj -> Det N_common_animate_iobj [0.5] | N_prop_iobj [0.5]
NP_animate_nsubj_targeted -> N_prop_nsubj_targeted [1.0]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [0.5] | Det N_common_inanimate_dobj PP_loc [0.5]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
AUX -> 'was' [1.0]
BY -> 'by' [1.0]
N_common_animate_dobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_animate_iobj -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_nsubj_targeted -> {target_item_str}
N_prop_nsubj -> {proper_nouns_str}
N_prop_dobj -> {proper_nouns_str}
N_prop_iobj -> {proper_nouns_str}
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
V_trans_omissible -> {V_trans_omissible_str}
V_trans_not_omissible -> {V_trans_not_omissible_str}
V_unacc -> {V_unacc_str}
V_cp_taking -> {V_cp_taking_str}
V_inf_taking -> {V_inf_taking_str}
V_unerg -> {V_unerg_str}
V_inf -> {V_inf_str}
V_dat -> {V_dat_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
INF -> 'to' [1.0]
""".format(animate_nouns_str=animate_nouns_str,
           inanimate_nouns_str=inanimate_nouns_str,
           proper_nouns_str=proper_nouns_str,
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,           
           V_trans_omissible_str=V_trans_omissible_str,
           V_trans_not_omissible_str=V_trans_not_omissible_str,
           V_cp_taking_str=V_cp_taking_str,
           V_inf_taking_str=V_inf_taking_str,
           V_unerg_str=V_unerg_str,
           V_inf_str=V_inf_str,
           V_dat_str=V_dat_str,
           V_unacc_str=V_unacc_str,
           target_item_str='{}'
          )


# # Subsets of main grammar that only generate target common noun/proper noun as objects

# In[ ]:


common_noun_object_grammar_str = """

S -> NP_animate_nsubj VP_external [1.0]
VP_external -> V_unacc NP_dobj [0.18] \
             | V_trans_omissible NP_dobj [0.18] | V_trans_not_omissible NP_dobj [0.18] \
             | V_dat NP_dobj PP_iobj [0.18] | V_dat NP_animate_iobj NP_dobj [0.18] \
             | V_cp_taking C S [0.1]
NP_dobj -> NP_animate_dobj [1.0]
NP_animate_dobj -> Det N_common_animate_dobj [0.5] | Det N_common_animate_dobj PP_loc [0.5]
NP_animate_iobj -> Det N_common_animate_iobj [0.5] | N_prop_iobj [0.5]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
AUX -> 'was' [1.0]
BY -> 'by' [1.0]
N_common_animate_dobj -> {target_item_str}
N_common_animate_iobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_iobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_trans_omissible -> {V_trans_omissible_str}
V_trans_not_omissible -> {V_trans_not_omissible_str}
V_cp_taking -> {V_cp_taking_str}
V_inf_taking -> {V_inf_taking_str}
V_unacc -> {V_unacc_str}
V_unerg -> {V_unerg_str}
V_inf -> {V_inf_str}
V_dat -> {V_dat_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
INF -> 'to' [1.0]
""".format(animate_nouns_str=animate_nouns_str,
           inanimate_nouns_str=inanimate_nouns_str,
           proper_nouns_str=proper_nouns_str,
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,            
           V_trans_omissible_str=V_trans_omissible_str,
           V_trans_not_omissible_str=V_trans_not_omissible_str, 
           V_cp_taking_str=V_cp_taking_str, 
           V_inf_taking_str=V_inf_taking_str, 
           V_unacc_str=V_unacc_str, 
           V_unerg_str=V_unerg_str, 
           V_inf_str=V_inf_str, 
           V_dat_str=V_dat_str, 
           target_item_str='{}'
          )

proper_noun_object_grammar_str = """

S -> NP_animate_nsubj VP_external [1.0]
VP_external -> V_unacc NP_dobj [0.18] \
             | V_trans_omissible NP_dobj [0.18] | V_trans_not_omissible NP_dobj [0.18] \
             | V_dat NP_dobj PP_iobj [0.18] | V_dat NP_animate_iobj NP_dobj [0.18] \
             | V_cp_taking C S [0.1] 
NP_dobj -> N_prop_dobj [1.0]
NP_animate_iobj -> Det N_common_animate_iobj [0.5] | N_prop_iobj [0.5]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.50]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
AUX -> 'was' [1.0]
BY -> 'by' [1.0]
N_common_animate_iobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_iobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
N_prop_dobj -> {target_item_str}
V_trans_omissible -> {V_trans_omissible_str}
V_trans_not_omissible -> {V_trans_not_omissible_str}
V_cp_taking -> {V_cp_taking_str}
V_inf_taking -> {V_inf_taking_str}
V_unacc -> {V_unacc_str}
V_unerg -> {V_unerg_str}
V_inf -> {V_inf_str}
V_dat -> {V_dat_str}
PP_iobj -> P NP_animate_iobj [1.0]
P -> 'to' [1.0]
INF -> 'to' [1.0]
""".format(animate_nouns_str=animate_nouns_str,
           inanimate_nouns_str=inanimate_nouns_str,
           proper_nouns_str=proper_nouns_str,
           V_trans_omissible_str=V_trans_omissible_str,
           V_trans_not_omissible_str=V_trans_not_omissible_str, 
           V_cp_taking_str=V_cp_taking_str, 
           V_inf_taking_str=V_inf_taking_str, 
           V_unacc_str=V_unacc_str, 
           V_unerg_str=V_unerg_str, 
           V_inf_str=V_inf_str, 
           V_dat_str=V_dat_str, 
           target_item_str='{}'
          )


# # Subset of main grammar that only generates target infinitival constructions

# In[ ]:


infinitival_verb_grammar_str = """

S -> NP_animate_nsubj VP_external [1.0] 
VP_external -> V_inf_taking INF V_inf [0.96] \
             | V_cp_taking C S [0.04]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_common_animate_nsubj -> {animate_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_cp_taking -> {V_cp_taking_str}
V_inf_taking -> {V_inf_taking_str}
V_inf -> {target_item_str}
INF -> 'to' [1.0]
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           V_cp_taking_str=V_cp_taking_str, 
           V_inf_taking_str=V_inf_taking_str, 
           target_item_str='{}'
          )


# # Subset of main grammar that generates a target transitive with D.O.

# In[ ]:


transitive_with_object_grammar_str = """

S -> NP_animate_nsubj VP_external [1.0]
VP_external -> V_trans NP_dobj [0.91] | | V_cp_taking C S [0.09]
NP_dobj -> NP_inanimate_dobj [0.5] | NP_animate_dobj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj [0.25] | Det N_common_animate_dobj PP_loc [0.25] \
                 | N_prop_dobj [0.50]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [0.5] | Det N_common_inanimate_dobj PP_loc [0.5]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_common_animate_dobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_dobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_cp_taking -> {V_cp_taking_str}
V_trans -> {target_item_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
""".format(animate_nouns_str=animate_nouns_str,
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,
           target_item_str='{}', 
           V_cp_taking_str=V_cp_taking_str
          )


# # Subsets of main grammar that only generate target transitives with animate subjects (i.e., agents)

# In[ ]:


transitive_with_animate_subject_grammar_str = """

S -> NP_animate_nsubj_targeted VP_external [0.9] | NP_animate_nsubj VP_CP [0.1]
VP_external -> V_trans_not_omissible_str NP_dobj [1.0]
VP_CP -> V_cp_taking C S [1.0]
NP_dobj -> NP_inanimate_dobj [0.5] | NP_animate_dobj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj [0.25] | Det N_common_animate_dobj PP_loc [0.25] \
                 | N_prop_dobj [0.50]
NP_animate_nsubj_targeted -> Det N_common_animate_nsubj_targeted [1.0]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [0.5] | Det N_common_inanimate_dobj PP_loc [0.5]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_common_animate_dobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_animate_nsubj_targeted -> {target_item_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_dobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_cp_taking -> {V_cp_taking_str}
V_trans_not_omissible_str -> {V_trans_not_omissible_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
""".format(animate_nouns_str=animate_nouns_str,
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,
           target_item_str='{}', 
           V_cp_taking_str=V_cp_taking_str,
           V_trans_not_omissible_str=V_trans_not_omissible_str
          )

unaccusative_with_animate_subject_grammar_str = """

S -> NP_animate_nsubj VP_external [0.05] | VP_internal [0.95]
VP_external -> V_cp_taking C S [1.0]
VP_internal -> NP_dobj V_unacc [1.0]
NP_dobj -> NP_animate_dobj [1.0]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj [1.0]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_common_animate_nsubj -> {animate_nouns_str}
N_common_animate_dobj -> {target_item_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_cp_taking -> {V_cp_taking_str}
V_unacc -> {V_unacc_str}
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           V_cp_taking_str=V_cp_taking_str, 
           V_unacc_str=V_unacc_str,
           target_item_str='{}'
          )


# # Grammar with high CP probability (for CP recursion cases; structural gen)

# In[ ]:


cp_high_prob_grammar_str = """

S -> NP_animate_nsubj VP_external [0.01] | VP_internal [0.01] \
   | NP_inanimate_nsubjpass VP_passive [0.01] | NP_animate_nsubjpass VP_passive_dat [0.01] \
   | NP_animate_nsubj VP_CP [0.96]
VP_CP -> V_cp_taking C S [1.0]
VP_external -> V_unerg [0.125] | V_unacc NP_dobj [0.125] \
             | V_trans_omissible [0.125] | V_trans_omissible NP_dobj [0.125] \
             | V_trans_not_omissible NP_dobj [0.125] | V_inf_taking INF V_inf [0.125] \
             | V_dat NP_inanimate_dobj PP_iobj [0.125] | V_dat NP_animate_iobj NP_inanimate_dobj [0.125]
VP_internal -> NP_unacc_subj V_unacc [1.0]
VP_passive -> AUX V_trans_not_omissible_pp [0.125] | AUX V_trans_not_omissible_pp BY NP_animate_nsubj [0.125] |  \
              AUX V_trans_omissible_pp [0.125] | AUX V_trans_omissible_pp BY NP_animate_nsubj [0.125] |  \
              AUX V_unacc_pp [0.125] | AUX V_unacc_pp BY NP_animate_nsubj [0.125] | \
              AUX V_dat_pp PP_iobj [0.125] | AUX V_dat_pp PP_iobj BY NP_animate_nsubj [0.125]
VP_passive_dat -> AUX V_dat_pp NP_inanimate_dobj [0.5] | AUX V_dat_pp NP_inanimate_dobj BY NP_animate_nsubj [0.5]
NP_dobj -> NP_inanimate_dobj [0.5] | NP_animate_dobj [0.5]
NP_unacc_subj -> NP_inanimate_dobj_noPP [0.5] | NP_animate_dobj_noPP [0.5]
NP_animate_dobj_noPP -> Det N_common_animate_dobj [0.5] | N_prop_dobj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj [0.25] | Det N_common_animate_dobj PP_loc [0.25] \
                 | N_prop_dobj [0.50]
NP_animate_iobj -> Det N_common_animate_iobj [0.5] | N_prop_iobj [0.5]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_animate_nsubjpass -> Det N_common_animate_nsubjpass [0.5] | N_prop_nsubjpass [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [0.5] | Det N_common_inanimate_dobj PP_loc [0.5]
NP_inanimate_dobj_noPP -> Det N_common_inanimate_dobj [1.0]
NP_inanimate_nsubjpass -> Det N_common_inanimate_nsubjpass [1.0]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
AUX -> 'was' [1.0]
BY -> 'by' [1.0]
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
N_common_animate_dobj -> {animate_nouns_str}
N_common_animate_iobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_animate_nsubjpass -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_common_inanimate_nsubjpass -> {inanimate_nouns_str}
N_prop_dobj -> {proper_nouns_str}
N_prop_iobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
N_prop_nsubjpass -> {proper_nouns_str}
V_trans_omissible -> {V_trans_omissible_str}
V_trans_omissible_pp -> {V_trans_omissible_pp_str}
V_trans_not_omissible -> {V_trans_not_omissible_str}
V_trans_not_omissible_pp -> {V_trans_not_omissible_pp_str}
V_cp_taking -> {V_cp_taking_str}
V_inf_taking -> {V_inf_taking_str}
V_unacc -> {V_unacc_str}
V_unacc_pp -> {V_unacc_pp_str}
V_unerg -> {V_unerg_str}
V_inf -> {V_inf_str}
V_dat -> {V_dat_str}
V_dat_pp -> {V_dat_pp_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
INF -> 'to' [1.0]
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,
           V_trans_omissible_str=V_trans_omissible_str, 
           V_trans_omissible_pp_str=V_trans_omissible_pp_str, 
           V_trans_not_omissible_str=V_trans_not_omissible_str, 
           V_trans_not_omissible_pp_str=V_trans_not_omissible_pp_str, 
           V_cp_taking_str=V_cp_taking_str, 
           V_inf_taking_str=V_inf_taking_str, 
           V_unacc_str=V_unacc_str, 
           V_unacc_pp_str=V_unacc_pp_str, 
           V_unerg_str=V_unerg_str, 
           V_inf_str=V_inf_str, 
           V_dat_str=V_dat_str, 
           V_dat_pp_str=V_dat_pp_str
          )


# # Grammar with high PP in object position (for PP recursion cases; structural gen)

# In[ ]:


obj_pp_high_prob_grammar_str = """

S -> NP_animate_nsubj VP_external [0.90] | NP_animate_nsubj VP_CP [0.10]
VP_CP -> V_cp_taking C S [1.0]
VP_external -> V_unacc NP_dobj [0.2] | V_trans_omissible NP_dobj [0.2] \
             | V_trans_not_omissible NP_dobj [0.2] \
             | V_dat NP_inanimate_dobj PP_iobj [0.2] | V_dat NP_animate_iobj NP_inanimate_dobj [0.2]
NP_dobj -> NP_inanimate_dobj [0.5] | NP_animate_dobj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj PP_loc [1.0] 
NP_animate_iobj -> Det N_common_animate_iobj [0.5] | N_prop_iobj [0.5]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj PP_loc [1.0]
NP_on -> Det N_on PP_loc [0.9] | Det N_on [0.1]
NP_in -> Det N_in PP_loc [0.9] | Det N_in [0.1]
NP_beside -> Det N_beside PP_loc [0.9] | Det N_beside [0.1]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
AUX -> 'was' [1.0]
BY -> 'by' [1.0]
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
N_common_animate_dobj -> {animate_nouns_str}
N_common_animate_iobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_dobj -> {proper_nouns_str}
N_prop_iobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_trans_omissible -> {V_trans_omissible_str}
V_trans_not_omissible -> {V_trans_not_omissible_str}
V_cp_taking -> {V_cp_taking_str}
V_unacc -> {V_unacc_str}
V_dat -> {V_dat_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,
           V_trans_omissible_str=V_trans_omissible_str, 
           V_trans_omissible_pp_str=V_trans_omissible_pp_str, 
           V_trans_not_omissible_str=V_trans_not_omissible_str, 
           V_trans_not_omissible_pp_str=V_trans_not_omissible_pp_str, 
           V_cp_taking_str=V_cp_taking_str, 
           V_inf_taking_str=V_inf_taking_str, 
           V_unacc_str=V_unacc_str, 
           V_unacc_pp_str=V_unacc_pp_str, 
           V_unerg_str=V_unerg_str, 
           V_inf_str=V_inf_str, 
           V_dat_str=V_dat_str, 
           V_dat_pp_str=V_dat_pp_str
          )


# # Subsets of main grammar that only generate target unaccusatives and causatives

# In[ ]:


unaccusative_grammar_str = """

S -> NP_animate_nsubj VP_external [0.1] | VP_internal [0.9]
VP_external -> V_cp_taking C S [1.0]
VP_internal -> NP_dobj V_unacc [1.0]
NP_dobj -> NP_inanimate_dobj [0.5] | NP_animate_dobj [0.5]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj [0.5] | N_prop_dobj [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [1.0]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_common_animate_nsubj -> {animate_nouns_str}
N_common_animate_dobj -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
N_prop_dobj -> {proper_nouns_str}
V_cp_taking -> {V_cp_taking_str}
V_unacc -> {target_item_str}
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           V_cp_taking_str=V_cp_taking_str, 
           target_item_str='{}'
          )

causative_grammar_str = """

S -> NP_animate_nsubj VP_external [1.0]
VP_external -> V_unacc NP_dobj [0.91] | V_cp_taking C S [0.09]
NP_dobj -> NP_inanimate_dobj [0.5] | NP_animate_dobj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj [0.25] | Det N_common_animate_dobj PP_loc [0.25] \
                 | N_prop_dobj [0.50]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [0.5] | Det N_common_inanimate_dobj PP_loc [0.5]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
N_common_animate_dobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_dobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_cp_taking -> {V_cp_taking_str}
V_unacc -> {target_item_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
""".format(animate_nouns_str=animate_nouns_str,
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,
           V_cp_taking_str=V_cp_taking_str, 
           target_item_str='{}'
          )


# # Subsets of main grammar that only generate target transitives with DO omission, target transitives with DO omission with animate subjects, and target unergatives

# In[ ]:


omitted_do_transitive_grammar_str = """

S -> NP_animate_nsubj VP_external [1.0]
VP_external -> V_trans_omissible [0.95] | V_cp_taking C S [0.05]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_common_animate_nsubj -> {animate_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_cp_taking -> {V_cp_taking_str}
V_trans_omissible -> {target_item_str}
""".format(animate_nouns_str=animate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           V_cp_taking_str=V_cp_taking_str, 
           target_item_str='{}'
          )

animate_subject_omitted_do_transitive_grammar_str = """

S -> NP_animate_nsubj_targeted VP_external [0.95] | NP_animate_nsubj VP_CP [0.05]
VP_external -> V_trans_omissible [1.0]
VP_CP -> V_cp_taking C S [1.0]
NP_animate_nsubj_targeted -> Det N_common_animate_nsubj_targeted [1.0]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_common_animate_nsubj_targeted -> {target_item_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_cp_taking -> {V_cp_taking_str}
V_trans_omissible -> {V_trans_omissible_str}
""".format(proper_nouns_str=proper_nouns_str, 
           animate_nouns_str=animate_nouns_str,
           V_cp_taking_str=V_cp_taking_str, 
           V_trans_omissible_str=V_trans_omissible_str,
           target_item_str='{}'
          )

animate_subject_unergative_grammar_str = """

S -> NP_animate_nsubj_targeted VP_external [0.95] | NP_animate_nsubj VP_CP [0.05]
VP_external -> V_unerg [1.0]
VP_CP -> V_cp_taking C S [1.0]
NP_animate_nsubj_targeted -> Det N_common_animate_nsubj_targeted [1.0]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_common_animate_nsubj_targeted -> {target_item_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_cp_taking -> {V_cp_taking_str}
V_unerg -> {V_unerg_str}
""".format(proper_nouns_str=proper_nouns_str, 
           animate_nouns_str=animate_nouns_str,
           V_cp_taking_str=V_cp_taking_str, 
           V_unerg_str=V_unerg_str,
           target_item_str='{}'
          )


# # Subset of main grammar that only generates target actives

# In[ ]:


active_grammar_str = """

S -> NP_animate_nsubj VP_external [1.0]
VP_external -> V_trans_not_omissible NP_dobj [0.9] | V_cp_taking C S [0.1]
NP_dobj -> NP_inanimate_dobj [0.5] | NP_animate_dobj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj [0.25] | Det N_common_animate_dobj PP_loc [0.25] \
                 | N_prop_dobj [0.50]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [0.5] | Det N_common_inanimate_dobj PP_loc [0.5]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
N_common_animate_dobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_dobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_trans_not_omissible -> {target_item_str}
V_cp_taking -> {V_cp_taking_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str,
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,
           V_cp_taking_str=V_cp_taking_str, 
           target_item_str='{}'
          )


# # Subsets of main grammar that only generate target passives

# In[ ]:


passive_grammar_str = """

S -> NP_animate_nsubj VP_external [0.05] | NP_inanimate_nsubjpass VP_passive [0.95]
VP_external -> V_cp_taking C S [1.0]
VP_passive -> AUX V_trans_not_omissible_pp [0.5] | AUX V_trans_not_omissible_pp BY NP_animate_nsubj [0.5] 
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_inanimate_nsubjpass -> Det N_common_inanimate_nsubjpass [1.0]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
AUX -> 'was' [1.0]
BY -> 'by' [1.0]
N_common_animate_nsubj -> {animate_nouns_str}
N_common_inanimate_nsubjpass -> {inanimate_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_trans_not_omissible_pp -> {target_item_str}
V_cp_taking -> {V_cp_taking_str}
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           target_item_str='{}',
           V_cp_taking_str=V_cp_taking_str
          )

passive_without_agent_grammar_str = """

S -> NP_animate_nsubj VP_external [0.03] | NP_inanimate_nsubjpass VP_passive [0.97]
VP_external -> V_cp_taking C S [1.0]
VP_passive -> AUX V_trans_not_omissible_pp [1.0]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_inanimate_nsubjpass -> Det N_common_inanimate_nsubjpass [1.0]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
AUX -> 'was' [1.0]
N_common_animate_nsubj -> {animate_nouns_str}
N_common_inanimate_nsubjpass -> {inanimate_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_trans_not_omissible_pp -> {target_item_str}
V_cp_taking -> {V_cp_taking_str}
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           target_item_str='{}', 
           V_cp_taking_str=V_cp_taking_str
          )


# # Subset of main grammar that only generates target double object datives

# In[ ]:


double_object_grammar_str = """

S -> NP_animate_nsubj VP_external [1.0]
VP_external -> V_cp_taking C S [0.1] \
             | V_dat NP_animate_iobj NP_inanimate_dobj [0.9]
NP_animate_iobj -> Det N_common_animate_iobj [0.5] | N_prop_iobj [0.5]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [0.5] | Det N_common_inanimate_dobj PP_loc [0.5]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_common_animate_iobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_iobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_cp_taking -> {V_cp_taking_str}
V_dat -> {target_item_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,
           V_cp_taking_str=V_cp_taking_str, 
           target_item_str="{}"
          )


# # Subset of main grammar that only generates target PP datives

# In[ ]:


PP_grammar_str = """

S -> NP_animate_nsubj VP_external [1.0] 
VP_external -> V_cp_taking C S [0.1] | V_dat NP_inanimate_dobj PP_iobj [0.9] 
NP_animate_iobj -> Det N_common_animate_iobj [0.5] | N_prop_iobj [0.5]
NP_animate_nsubj -> Det N_common_animate_nsubj [0.5] | N_prop_nsubj [0.5]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [0.5] | Det N_common_inanimate_dobj PP_loc [0.5]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
N_common_animate_iobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_prop_iobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
V_cp_taking -> {V_cp_taking_str}
V_dat -> {target_item_str}
INF -> 'to' [1.0]
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,
           V_cp_taking_str=V_cp_taking_str, 
           target_item_str="{}"
          )


# # (Non-subset) of main grammar that generates NP PP constructions in the subject position (structural gen)

# In[ ]:


subj_NP_PP_grammar_str = """

S -> NP_animate_nsubj VP_external [0.55] | VP_internal [0.05] \
   | NP_inanimate_nsubjpass VP_passive [0.20] | NP_animate_nsubjpass VP_passive_dat [0.20]
VP_external -> V_unerg [0.0875] | V_unacc NP_dobj [0.0875] \
             | V_trans_omissible [0.0875] | V_trans_omissible NP_dobj [0.0875] \
             | V_trans_not_omissible NP_dobj [0.0875] | V_inf_taking INF V_inf [0.0875] \
             | V_cp_taking C S [0.30] \
             | V_dat NP_inanimate_dobj PP_iobj [0.0875] | V_dat NP_animate_iobj NP_inanimate_dobj [0.0875]
VP_internal -> NP_unacc_subj V_unacc [1.0]
VP_passive -> AUX V_trans_not_omissible_pp [0.125] | AUX V_trans_not_omissible_pp BY NP_animate_nsubj [0.125] |  \
              AUX V_trans_omissible_pp [0.125] | AUX V_trans_omissible_pp BY NP_animate_nsubj [0.125] |  \
              AUX V_unacc_pp [0.125] | AUX V_unacc_pp BY NP_animate_nsubj [0.125] | \
              AUX V_dat_pp PP_iobj [0.125] | AUX V_dat_pp PP_iobj BY NP_animate_nsubj [0.125]
VP_passive_dat -> AUX V_dat_pp NP_inanimate_dobj [0.5] | AUX V_dat_pp NP_inanimate_dobj BY NP_animate_nsubj [0.5]
NP_dobj -> NP_inanimate_dobj [0.5] | NP_animate_dobj [0.5]
NP_unacc_subj -> NP_inanimate_nsubj [0.5] | NP_animate_nsubj [0.5]
NP_animate_dobj -> Det N_common_animate_dobj [0.25] | Det N_common_animate_dobj PP_loc [0.25] \
                 | N_prop_dobj [0.50]
NP_animate_iobj -> Det N_common_animate_iobj [0.5] | N_prop_iobj [0.5]
NP_animate_nsubj -> Det N_common_animate_nsubj PP_loc [1.0]
NP_animate_nsubjpass -> Det N_common_animate_nsubjpass PP_loc [1.0]
NP_inanimate_dobj -> Det N_common_inanimate_dobj [0.5] | Det N_common_inanimate_dobj PP_loc [0.5]
NP_inanimate_nsubj -> Det N_common_inanimate_dobj PP_loc [1.0]
NP_inanimate_nsubjpass -> Det N_common_inanimate_nsubjpass PP_loc [1.0]
NP_on -> Det N_on PP_loc [0.1] | Det N_on [0.9]
NP_in -> Det N_in PP_loc [0.1] | Det N_in [0.9]
NP_beside -> Det N_beside PP_loc [0.1] | Det N_beside [0.9]
Det -> 'the' [0.5] | 'a' [0.5]
C -> 'that' [1.0]
AUX -> 'was' [1.0]
BY -> 'by' [1.0]
N_common_animate_dobj -> {animate_nouns_str}
N_common_animate_iobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_animate_nsubjpass -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_common_inanimate_nsubjpass -> {inanimate_nouns_str}
N_prop_dobj -> {proper_nouns_str}
N_prop_iobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
N_prop_nsubjpass -> {proper_nouns_str}
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
V_trans_omissible -> {V_trans_omissible_str}
V_trans_omissible_pp -> {V_trans_omissible_pp_str}
V_trans_not_omissible -> {V_trans_not_omissible_str}
V_trans_not_omissible_pp -> {V_trans_not_omissible_pp_str}
V_cp_taking -> {V_cp_taking_str}
V_inf_taking -> {V_inf_taking_str}
V_unacc -> {V_unacc_str}
V_unacc_pp -> {V_unacc_pp_str}
V_unerg -> {V_unerg_str}
V_inf -> {V_inf_str}
V_dat -> {V_dat_str}
V_dat_pp -> {V_dat_pp_str}
PP_iobj -> P_iobj NP_animate_iobj [1.0]
PP_loc -> P_on NP_on [0.333] | P_in NP_in [0.333] | P_beside NP_beside [0.334]
P_iobj -> 'to' [1.0]
P_on -> 'on' [1.0]
P_in -> 'in' [1.0]
P_beside -> 'beside' [1.0]
INF -> 'to' [1.0]
""".format(animate_nouns_str=animate_nouns_str, 
           inanimate_nouns_str=inanimate_nouns_str, 
           proper_nouns_str=proper_nouns_str, 
           in_nouns_str=in_nouns_str,
           on_nouns_str=on_nouns_str,
           beside_nouns_str=beside_nouns_str,
           V_trans_omissible_str=V_trans_omissible_str, 
           V_trans_omissible_pp_str=V_trans_omissible_pp_str, 
           V_trans_not_omissible_str=V_trans_not_omissible_str, 
           V_trans_not_omissible_pp_str=V_trans_not_omissible_pp_str, 
           V_cp_taking_str=V_cp_taking_str, 
           V_inf_taking_str=V_inf_taking_str, 
           V_unacc_str=V_unacc_str, 
           V_unacc_pp_str=V_unacc_pp_str, 
           V_unerg_str=V_unerg_str, 
           V_inf_str=V_inf_str, 
           V_dat_str=V_dat_str,
           V_dat_pp_str=V_dat_pp_str
          )


# # Fill in target lexical items for full grammar

# In[ ]:


random.seed(333)

# Grammar for train/dev/test
main_grammar = PCFG.fromstring(main_grammar_str)

# See only as subject, generalize to unseen cases as subjects and objects
hedgehog_subject_grammar = PCFG.fromstring(common_noun_subject_grammar_str.format("'hedgehog' [1.0]"))
hedgehog_object_grammar = PCFG.fromstring(common_noun_object_grammar_str.format("'hedgehog' [1.0]"))
lina_subject_grammar = PCFG.fromstring(proper_noun_subject_grammar_str.format("'Lina' [1.0]"))
lina_object_grammar = PCFG.fromstring(proper_noun_object_grammar_str.format("'Lina' [1.0]"))

# See only object, generalize to unseen cases as subjects and objects
cockroach_object_grammar = PCFG.fromstring(common_noun_object_grammar_str.format("'cockroach' [1.0]"))
cockroach_subject_grammar = PCFG.fromstring(common_noun_subject_grammar_str.format("'cockroach' [1.0]"))

charlie_object_grammar = PCFG.fromstring(proper_noun_object_grammar_str.format("'Charlie' [1.0]"))
charlie_subject_grammar = PCFG.fromstring(proper_noun_subject_grammar_str.format("'Charlie' [1.0]"))

# See only primitive form of common noun, generalize to subjects and objects
shark_subject_grammar = PCFG.fromstring(common_noun_subject_grammar_str.format("'shark' [1.0]"))
shark_object_grammar = PCFG.fromstring(common_noun_object_grammar_str.format("'shark' [1.0]"))

# See only primitive form of proper noun, generalize to subjects and objects
paula_subject_grammar = PCFG.fromstring(proper_noun_subject_grammar_str.format("'Paula' [1.0]"))
paula_object_grammar = PCFG.fromstring(proper_noun_object_grammar_str.format("'Paula' [1.0]"))

# See only primitive form of verb, generalize to infinitival arguments
crawl_infinitival_grammar = PCFG.fromstring(infinitival_verb_grammar_str.format("'crawl' [1.0]"))

# See only unaccusative form of verb, generalize to transitive
shattered_unaccusative_grammar = PCFG.fromstring(unaccusative_grammar_str.format("'shattered' [1.0]"))
shattered_transitive_grammar = PCFG.fromstring(causative_grammar_str.format("'shattered' [1.0]"))

# See only object omissible transitive form of verb, generalize to transitive
baked_omitted_do_transitive_grammar = PCFG.fromstring(omitted_do_transitive_grammar_str.format("'baked' [1.0]"))
baked_transitive_with_object_grammar = PCFG.fromstring(transitive_with_object_grammar_str.format("'baked' [1.0]"))

# Take an NP that only appears as an animate subject of transitive verbs.
# Generalize to subject of a known unaccusative verb.
cobra_transitive_with_animate_subject_grammar = PCFG.fromstring(transitive_with_animate_subject_grammar_str.format("'cobra' [1.0]"))
unaccusative_with_animate_subject_grammar = PCFG.fromstring(unaccusative_with_animate_subject_grammar_str.format("'cobra' [1.0]"))

# Take an NP that only appears as an animate subject of unaccusative verbs.
# Generalize to subject of a known omissible transitive.
hippo_unaccusative_with_animate_subject_grammar = PCFG.fromstring(unaccusative_with_animate_subject_grammar_str.format("'hippo' [1.0]"))
ate_omitted_do_transitive_animate_subject_grammar = PCFG.fromstring(animate_subject_omitted_do_transitive_grammar_str.format("'hippo' [1.0]"))

# See only actives, generalize to passives
blessed_active_grammar = PCFG.fromstring(active_grammar_str.format("'blessed' [1.0]"))
blessed_passive_grammar = PCFG.fromstring(passive_grammar_str.format("'blessed' [1.0]"))

# See only passives, generalize to actives
squeezed_passive_without_agent_grammar = PCFG.fromstring(passive_without_agent_grammar_str.format("'squeezed' [1.0]", "'squeezed' [1.0]"))
squeezed_active_grammar = PCFG.fromstring(active_grammar_str.format("'squeezed' [1.0]"))

# See only double object datives, generalize to PP datives
double_object_train_grammar = PCFG.fromstring(double_object_grammar_str.format("'teleported' [1.0]"))
PP_gen_grammar = PCFG.fromstring(PP_grammar_str.format("'teleported' [1.0]"))

# See only PP datives, generalize to double object datives
PP_train_grammar = PCFG.fromstring(PP_grammar_str.format("'shipped' [1.0]"))
double_object_gen_grammar = PCFG.fromstring(double_object_grammar_str.format("'shipped' [1.0]"))

# See only PP modifying object NPs, generalize to PP modifying subject NPs (both active & passive)
subj_NP_PP_grammar = PCFG.fromstring(subj_NP_PP_grammar_str)

# Given the training data (only contains depth 1 and 2 recursion of CPs), generalize to greater depths (3<=x<=12)
cp_high_prob_grammar = PCFG.fromstring(cp_high_prob_grammar_str)

# Given the training data (only contains depth 1 and 2 recursion of object NP PPs), generalize to greater depths (3<=x<=12)
obj_pp_high_prob_grammar = PCFG.fromstring(obj_pp_high_prob_grammar_str)

# grammar.start()
# grammar.productions()

candidate_grammars = [hedgehog_subject_grammar, hedgehog_object_grammar, lina_subject_grammar, lina_object_grammar, cockroach_object_grammar, cockroach_subject_grammar, charlie_object_grammar, charlie_subject_grammar, shark_subject_grammar, shark_object_grammar, paula_subject_grammar, paula_object_grammar, crawl_infinitival_grammar, shattered_unaccusative_grammar, shattered_transitive_grammar, baked_omitted_do_transitive_grammar, baked_transitive_with_object_grammar, cobra_transitive_with_animate_subject_grammar, unaccusative_with_animate_subject_grammar, hippo_unaccusative_with_animate_subject_grammar, ate_omitted_do_transitive_animate_subject_grammar, blessed_active_grammar, blessed_passive_grammar, squeezed_passive_without_agent_grammar, squeezed_active_grammar, double_object_train_grammar, PP_gen_grammar, PP_train_grammar, double_object_gen_grammar, subj_NP_PP_grammar, cp_high_prob_grammar, obj_pp_high_prob_grammar]

if __name__ == "__main__":
    parser = nltk.parse.pchart.InsideChartParser(main_grammar)

    candidate_parsers = [nltk.parse.pchart.InsideChartParser(cg) for cg in candidate_grammars]

    tokens = ['a', 'landlord', 'hoped', 'that', 'the', 'hippo', 'decomposed']
    for c_parser in candidate_parsers:
      try:
        parses = list(c_parser.parse(tokens))
      except ValueError:
        continue
      if len(parses) > 0:
        print("Passed")

    tokens = ['the', 'creature', 'grew', 'Charlie']
    for c_parser in candidate_parsers:
      try:
        parses = list(c_parser.parse(tokens))
      except ValueError:
        continue
      if len(parses) > 0:
        print("Passed")

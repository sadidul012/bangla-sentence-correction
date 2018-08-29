from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.stats as stat
import re

import collections
import json
import numpy as np
import process

with open('config.json') as f:
    config = json.load(f)

save_dir = config["save_dir"]
data_dir = config["data_dir"]
file_name = config["file_name"]

lines = []
vocabulary = []
vocabulary_size = config["vocabulary_size"]
report_point = config["report_point"]

i = 0

with open(data_dir+file_name, "r") as file:
    for line in file:
        if i > config["max_line_read"]:
            break
        wss = line.strip()
        ws = wss.split(" ")

        vocabulary.extend(ws)
        lines.append(ws)

        i += 1
        if i % report_point == 0:
            print("{} lines processed".format(i))

print('Total lines', i)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.


def build_data_set(words, n_words):
    """Process raw inputs into a bangla-text1."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())
    # print(count)
    c = np.transpose(count)
    c = np.array(c[1], dtype=np.int32)
    print("\nUnique words", len(count))
    print("Count mode", stat.mode(c))
    print("Count average", np.average(c))

    dictionary = dict()
    dictionary["UNK"] = len(dictionary)
    dictionary["END"] = len(dictionary)
    for word, c in count:
        if (config["min_count"] <= c <= config["max_count"]) and len(dictionary)<n_words:
            dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for line in lines:
        line_data = list()
        for word in line:
            index = dictionary.get(word, 0)
            if index == 0:  # dictionary['UNK']
                unk_count += 1
            line_data.append(index)
        data.append(line_data)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_data_set(vocabulary, vocabulary_size)

# print(data)
del vocabulary  # Hint to reduce memory.
print('\nMost common words (+UNK)', count[:50])
print("Dictionary", [(i, reverse_dictionary[i]) for i in range(10)])
print("Vocab size: ", len(reverse_dictionary), "out of", config["vocabulary_size"])
print('\nSample data')
for d in data[:30]:
    print(d, [reverse_dictionary[i] for i in d])


train_file = save_dir+file_name
open(train_file, 'w').close()

with open(train_file, "a") as file:
    for d in data:
        line = " ".join([str(dn) for dn in d]) + "\n"
        file.write(line)

train_file = save_dir+"human_"+file_name
open(train_file, 'w').close()

with open(train_file, "a") as file:
    for d in lines:
        line = " ".join([str(dn) for dn in d]) + "\n"
        file.write(line)

train_file = save_dir+"human_UNK_"+file_name
open(train_file, 'w').close()

with open(train_file, "a") as file:
    for d in lines:
        line = []
        for dn in d:
            if dn in dictionary.keys():
                line.append(dn)
            else:
                line.append("UNK")

        line = " ".join(line) + "\n"
        file.write(line)

process.save_obj(reverse_dictionary, "reverse_dictionary")

process.save_obj(dictionary, "dictionary")
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os.path as osp
import json
import nltk
import sys

nltk_tags = ["$", "--", ",", ".", "''", "(", ")", "``", "CC", "CD", "DT", "EX", "FW", "IN", "JJ",
                 "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$",
                 "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
                 "WDT", "WP", "WP$", "WRB", ":"]

nouns = ["NN", "NNP", "NNPS", "NNS"]
verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
adjectives = ["JJ", "JJR", "JJS"]
adjectivesJJ = ["JJ"]
adjectivesJJR = ["JJR"]
adjectivesJJS = ["JJS"]
numbers = ["CD"]
adverbs = ["RB", "RBR", "RBS"]
determiners = ["DT"]
prepositions = ["IN"]
particles = ["RP"]
pronouns = ["PRP", "PRP$"]

def parsePos(pos, tagsDict):
    phraseDict = {}
    for t in nltk_tags:
        phraseDict[t] = []

    for word, tag in pos:
        word = word.lower()
        if word not in tagsDict[tag]:
            tagsDict[tag][word] = 0
        tagsDict[tag][word] += 1
        phraseDict[tag].append(word)

    return tagsDict, phraseDict

def updateVocab(words, vocab):
    for word in words:
        word = word.lower()
        if word not in vocab:
            vocab[word] = 0
        vocab[word] += 1

    return vocab

def parseQueries(queryFile):
    tagsDict = {} # words per POS tag
    for t in nltk_tags:
        tagsDict[t] = {}

    vocab = {} # all words

    of = open(queryFile, 'r')
    queries = json.load(of)['results']
    query2TagDict = {} # words per video

    for q in queries:
        qs = queries[q]['sentences']
        vid_id = q
        query2TagDict[vid_id] = []
        for query in qs:
            if query == "": break
            # clean punctuation
            query = query.replace(',', ' ')
            query = query.replace('.', ' ')
            query = query.replace(':', ' ')
            query = query.replace(';', ' ')
            query = query.replace('!', ' ')
            query = query.replace('?', ' ')
            query = query.replace('"', ' ')
            query = query.replace('&', ' and ')
            query = query.replace('@', ' ')
            query = query.replace('(', ' ')
            query = query.replace(')', ' ')
            query = query.replace('[', ' ')
            query = query.replace(']', ' ')
            query = query.replace('<', ' ')
            query = query.replace('>', ' ')
            query = query.replace('`', ' ')
            query = query.replace('#', ' ')
            query = query.replace(u'\u2019', "'")
            #print(query)
            #
            words = query.split()
            vocab = updateVocab(words, vocab)
            #
            pos = nltk.pos_tag(nltk.word_tokenize(query))
            #if "'" in query:
            #    pass
            #    #print(query)
            #elif(len(pos) != len(query.split())):
            #    pass
            #    #print(query)
            tagsDict, phraseDict = parsePos(pos, tagsDict)
            query2TagDict[vid_id].append({'query': query, 'phraseDict': phraseDict})
    return tagsDict, query2TagDict, vocab

if __name__ == '__main__':

    # may need: nltk.download('averaged_perceptron_tagger')

    #nltk.download('averaged_perceptron_tagger')
    captionsPath = '/data2/activity_net/captions/'
    fileName = 'val_1.json'
    print(sys.argv[1])
    tagsDict, query2TagDict, vocab = parseQueries(sys.argv[1])
    pos_stats = {}
    pos_stats['tagsDict'] = tagsDict
    pos_stats['query2TagDict'] = query2TagDict
    pos_stats['vocab'] = vocab
    json.dump(pos_stats, open(fileName + '-pos.json', 'w'))

    # save some tags
    for tag in ['NN', 'NNS', 'PRP', 'VB']:
        fout = open('vocab.%s.tsv' % tag, 'w')
        for a, b in tagsDict[tag].items(): fout.write('%s\t%d\n' % (a.encode('utf-8'), b))
        fout.close()

    # save full vocab
    fout = open('vocab.tsv', 'w')
    for a, b in vocab.items(): fout.write('%s\t%d\n' % (a.encode('utf-8'), b))
    fout.close()

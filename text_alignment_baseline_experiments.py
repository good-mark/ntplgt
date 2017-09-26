#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Plagiarism detection for near-duplicate plagiarism.
"""

import os
import string
import hashlib
import sys
import sqlite3
import xml.dom.minidom
import codecs
from nltk.corpus import stopwords
from collections import deque
codecs.register(lambda name: codecs.lookup('utf-8') if name == 'cp65001' else None)

# Const
# =====

DELETECHARS = ''.join([string.punctuation, string.whitespace, string.digits, u'—«»…'])

SHINGLE_LEN = 3
SHINGLE_LEN_MINUS_1 = SHINGLE_LEN - 1


stop_words = stopwords.words('russian')
stop_words.extend([u'свой', u'своя', u'свое', u'свои', u'своего', u'своем', u'своему', u'своей', u'свою', u'своим', u'своих', u'своими'])
#for sw in stop_words:
#    print sw

clusters_db = sqlite3.connect('.\\lihonosov_clusters_full.sqlite')
clusters_cur = clusters_db.cursor()

# Helper functions
# ================

""" The following functions are some simple helper functions you can utilize
and modify to fit your own program.
"""

def print_array(array):
    for r in array:
        print r[0], r[1], r[2]

''' 
Проверяет, пересекаются ли два отрезка
'''
def check_intersection(first, second):
    if first[1] >= second[0] and first[0] < second[1] \
        and first[3] >= second[2] and first[2] < second[3]:
        #print '!', first, '!', second
        return True
    return False

def find_max_fragments(positions): #positions: set((susp_word_id, src_word_id, susp_start_pos, susp_end_pos, src_start_pos, src_end_pos))
    if len(positions) == 0:
        return []
    if len(positions) == 1:
        return [(1, 3, 1, 3)]
    #print 'after sorting', ' '.join([str(p) for p in ppositions])
    
    begin_element = positions[0]
    cur_element = positions[0]
    result = []
    for element in positions[1:]:
        # +1 - защита от вставки слов.
        if element[0] - cur_element[0] > SHINGLE_LEN + 1 or element[1] - cur_element[1] > SHINGLE_LEN + 1 \
                    or element[0] - cur_element[0] < 0 or element[1] - cur_element[1] < 0:
            if cur_element != begin_element:
                
                if cur_element[3] - begin_element[2] > 20:
                        if len(result) != 0 and check_intersection(result[-1], (begin_element[2], cur_element[3], begin_element[4], cur_element[5])):
                            '''
                            if cur_element[3] - result[-1][0] < 0 or cur_element[5] - result[-1][2] <0:
                                print result[-1][0], cur_element[3], result[-1][2], cur_element[5]
                            '''
                            result[-1] = (result[-1][0], cur_element[3], result[-1][2], cur_element[5])
                        else: 
                            result.append((begin_element[2], cur_element[3], begin_element[4], cur_element[5]))

            begin_element = element
            cur_element = element
            continue
        if element[0] - cur_element[0] <= SHINGLE_LEN + 1 and element[1] - cur_element[1] <= SHINGLE_LEN + 1:
            cur_element = element
            continue

    if len(result) != 0 and check_intersection(result[-1], (begin_element[2], cur_element[3], begin_element[4], cur_element[5])):
        '''
        if cur_element[3] - result[-1][0] < 0 or cur_element[5] - result[-1][2] <0:
            print result[-1][0], cur_element[3], result[-1][2], cur_element[5]
        '''
        result[-1] = (result[-1][0], cur_element[3], result[-1][2], cur_element[5])
    else: 
        result.append((begin_element[2], cur_element[3], begin_element[4], cur_element[5]))

    return result

def find_label(word):
    key_str = word
    clusters_cur.execute('SELECT class_id FROM clusters WHERE word=(?)', (key_str, ))
    result = clusters_cur.fetchone()
    return result
        

def get_label(word, mode=1):
    if mode == 1:
        return word
    else:
        label = find_label(word)
        #print word, label
        if label != None:            
            label = str(label[0])
            return label
        else: 
            #print word, 'not found in dict!'
            return '100000'

def replace_with_labels(array_of_filtered_words):
    print_array([(get_label(x[0], 1), x[1], x[2]) for x in array_of_filtered_words])
    return [(get_label(x[0], 1), x[1], x[2]) for x in array_of_filtered_words]


def tokenize(text):
    """ Tokenize a given text and return a dict containing all start and end
    positions for each token.
    Characters defined in the global string DELETECHARS will be ignored.

    Keyword arguments:
    text   -- the text to tokenize
    length -- the length of each token
    """
    tokens_with_idxs = []
    token = ""
    cur_start_idx = 0

    #Read one-by-one symbol
    for i in range(0, len(text)):
        if text[i] not in DELETECHARS:
            token += text[i]
        else:
            if cur_start_idx != i:
                #print token.lower(), cur_start_idx
                tokens_with_idxs.append((token.lower(), cur_start_idx, i - 1)) #token, start_pos, end_pos
                cur_start_idx = i + 1
                token = ""
            else:
                cur_start_idx = i + 1
    #print_array(tokens_with_idxs)
    return tokens_with_idxs

''' 
Предобрабатывает входящий текст: убирает стоп-слова. 
Возвращает последовательность слов входящего текста без стоп-слов.
''' 
def canonize(tokens):
    #or '"' in x or x == '-' or x == '--' or '\'' in x or '`' in x
    result = [(x[0], x[1], x[2]) for x in tokens if x[0] and len(x[0]) > 1 and (x[0] not in stop_words)]
    print_array(result)
    return result


def serialize_features(susp, src, features, outdir):
    """ Serialze a feature list into a xml file.
    The xml is structured as described in the readme file of the 
    PAN plagiarism corpus 2012. The filename will follow the naming scheme
    {susp}-{src}.xml and is located in the current directory.
    Existing files will be overwritten.

    Keyword arguments:
    susp     -- the filename of the suspicious document
    src      -- the filename of the source document
    features -- a list containing feature-tuples of the form
                ((start_pos_susp, end_pos_susp),
                 (start_pos_src, end_pos_src))
    """
    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, 'document', None)
    root = doc.documentElement
    root.setAttribute('reference', susp)
    doc.createElement('feature')

    for f in features:
        #print f
        feature = doc.createElement('feature')
        feature.setAttribute('name', 'detected-plagiarism')
        feature.setAttribute('this_offset', str(f[0]))
        feature.setAttribute('this_length', str(f[1] - f[0]))
        feature.setAttribute('source_reference', src)
        feature.setAttribute('source_offset', str(f[2]))
        feature.setAttribute('source_length', str(f[3] - f[2]))
        root.appendChild(feature)

    doc.writexml(open(outdir + susp.split('.')[0] + '-'
                      + src.split('.')[0] + '.xml', 'w'),
                 encoding='utf-8')



''' 
Для входящей последовательности слов генерирует шингл, возвращает множество хэшей от шингла с соответствующими индексами: 
номером слова в очищенном от стоп-слов тексте и номером начальной позиции первого символа. 
''' 
def generate_shingles(labels): #labels: [(label, start_pos, end_pos)]
    out = {}
    first = [0]
    first.extend([x[0] for x in labels[:SHINGLE_LEN_MINUS_1]])
    #print first
    seq = deque(first)
    for idx, word_with_id in enumerate(labels[SHINGLE_LEN_MINUS_1:]):
        seq.popleft()
        seq.append(word_with_id[0])
        sorted_seq = sorted(seq)
        seqhash = hashlib.md5( (' '.join(sorted_seq)).encode('utf-8') ).hexdigest()

        if seqhash not in out:
            out[seqhash] = []
        #print 'sh', idx, labels[idx][1], word_with_id[2]
        out[seqhash].append( (idx, labels[idx][1], word_with_id[2]) ) #return {shingle_hash: (word_id, start_pos, end_pos)}

    return out

def generate_shingles_array(labels): #labels: [(label, start_pos, end_pos)]
    out = []
    first = [0]
    first.extend([x[0] for x in labels[:SHINGLE_LEN_MINUS_1]])
    seq = deque(first)
    for idx, word_with_id in enumerate(labels[SHINGLE_LEN_MINUS_1:]):
        seq.popleft()
        seq.append(word_with_id[0])
        sorted_seq = sorted(seq)
        seqhash = hashlib.md5( (' '.join(sorted_seq)).encode('utf-8') ).hexdigest()
        #print 'sh', seqhash, labels[idx][1], word_with_id[2]
        out.append( (seqhash, labels[idx][1], word_with_id[2]) ) #return [(shingle_hash, start_pos, end_pos)]

    return out


# Plagiarism pipeline
# ===================

""" The following class implement a very basic baseline comparison, which
aims at near duplicate plagiarism. It is only intended to show a simple
pipeline your plagiarism detector can follow.
Replace the single steps with your implementation to get started.
"""

class Baseline:
    def __init__(self, susp, src, outdir):
        self.susp = susp
        self.src = src
        self.susp_file = os.path.split(susp)[1]
        self.src_file = os.path.split(src)[1]
        self.susp_id = os.path.splitext(susp)[0]
        self.src_id = os.path.splitext(src)[0]
        self.output = self.susp_id + '-' + self.src_id + '.xml'
        self.detections = None
	self.outdir=outdir

    def process(self):
        """ Process the plagiarism pipeline. """
        # if not os.path.exists(self.output):
        #    ...
        self.preprocess()
        self.detections = self.compare()
        self.postprocess()

    def preprocess(self):
        """ Preprocess the suspicious and source document. """
        # TODO: Implement your preprocessing steps here.
        susp_fp = codecs.open(self.susp, 'r', 'utf-8')
        self.susp_text = susp_fp.read()
        self.susp_tokens = tokenize(self.susp_text)
        self.susp_tokens_without_stopwords = canonize(self.susp_tokens)
        self.susp_labels = replace_with_labels(self.susp_tokens_without_stopwords)
        self.susp_shingles = generate_shingles_array(self.susp_labels)
        susp_fp.close()

        src_fp = codecs.open(self.src, 'r', 'utf-8')
        self.src_text = src_fp.read()
        self.src_tokens = tokenize(self.src_text)
        self.src_tokens_without_stopwords = canonize(self.src_tokens)
        self.src_labels = replace_with_labels(self.src_tokens_without_stopwords)
        self.src_shingles = generate_shingles(self.src_labels)
        src_fp.close()

    def compare(self):
        detections = []
        #Для каждого шингла в анализируемом документе ищем его среди шинглов документа-источника:
        for idx, shingle in enumerate(self.susp_shingles): 
            #Если такого шингла нет в источнике, переходим к следующему
            if shingle[0] not in self.src_shingles:
                continue

            result = self.src_shingles[shingle[0]]
            for r in result:   
                word_id = r[0]
                src_start_pos = r[1]
                src_end_pos = r[2]
                susp_start_pos = shingle[1]
                susp_end_pos = shingle[2]   
                if susp_end_pos - susp_start_pos < 0:
                    print '==============================================='
                print 'detection', idx, word_id, susp_start_pos, susp_end_pos, src_start_pos, src_end_pos
                detections.append((idx, word_id, susp_start_pos, susp_end_pos, src_start_pos, src_end_pos))

        fragments = find_max_fragments(detections)

        
        print 'right answers:'
        print self.susp_text[7385:7460], "!!!", self.src_text[4794:4886]
        #print self.susp_text[7541:7600], "!!!", self.src_text[4887:4941]
        print self.susp_text[7601:7836], "!!!", self.src_text[4942:5187]
        
        
        print 'my answers:'
        for f in fragments:
            print self.susp_text[f[0]:f[1]+1], "!!!", self.src_text[f[2]:f[3]+1]
        

        return fragments
               

    def postprocess(self):
        """ Postprocess the results. """
        # TODO: Implement your postprocessing steps here.
        serialize_features(self.susp_file, self.src_file, self.detections, self.outdir)

# Main
# ====

if __name__ == "__main__":
    """ Process the commandline arguments. We expect three arguments: The path
    pointing to the pairs file and the paths pointing to the directories where
    the actual source and suspicious documents are located.
    """
    if len(sys.argv) == 5:
        pairs = sys.argv[1]
        srcdir = sys.argv[2]
        suspdir = sys.argv[3]
        outdir = sys.argv[4]
        if outdir[-1] != "/":
            outdir+="/"
        lines = open(pairs, 'r').readlines()
        for line in lines:
            susp, src = line.split()
            print line
            baseline = Baseline(os.path.join(suspdir, susp),
                                os.path.join(srcdir, src), outdir)
            baseline.process()
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./pan12-plagiarism-text-alignment-example.py {pairs} {src-dir} {susp-dir} {out-dir}"]))

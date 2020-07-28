import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin
import pandas as pd
import numpy as np
from pathlib import Path
from os import listdir
import torch
from multiprocess import Pool

from others.logging import logger
from others.tokenization import BertTokenizer
from pytorch_transformers import XLNetTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]
count = 0


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)


def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])
            continue
        if (flag):
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt


def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused4]'] + byline + ['[unused5]']] + paras
        else:
            paras = [title + ['[unused4]']] + paras

        return paras, abs
    else:
        return None, None


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        if (args.bert_model == 'bert-base-multilingual-cased'):
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
            print(len(self.tokenizer.vocab))
            if (len(self.tokenizer.vocab) == 31748):
                f = open(args.bert_model + "/vocab.txt", "a")
                f.write("\n[unused1]\n[unused2]\n[unused3]\n[unused4]\n[unused5]\n[unused6]\n[unused7]")
                f.close()
                self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
            print(len(self.tokenizer.vocab))
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused1]'
        self.tgt_eos = '[unused2]'
        self.tgt_sent_split = '[unused3]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text, use_bert_basic_tokenizer)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused1] ' + ' [unused3] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused2]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']

        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(line)
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        rl = f.split('/')
        length = len(rl)
        real_name = rl[length - 1].split('.')[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)
        else:
            train_files.append(f)
    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset, ensure_ascii=False))
                    p_ct += 1
                    dataset = []
        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset, ensure_ascii=False))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    print(f)
    source, tgt = load_json(f, args.lower)
    return {'src': source, 'tgt': tgt}


def format_xsum_to_lines(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'valid']

    corpus_mapping = json.load(open(pjoin(args.raw_path, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')))

    for corpus_type in datasets:
        mapped_fnames = corpus_mapping[corpus_type]
        root_src = pjoin(args.raw_path, 'restbody')
        root_tgt = pjoin(args.raw_path, 'firstsentence')
        # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
        realnames = mapped_fnames

        a_lst = [(root_src, root_tgt, n) for n in realnames]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
            if (d is None):
                continue
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset, ensure_ascii=False))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset, ensure_ascii=False))
                p_ct += 1
                dataset = []


def _format_xsum_to_lines(params):
    src_path, root_tgt, name = params
    f_src = pjoin(src_path, name + '.restbody')
    f_tgt = pjoin(root_tgt, name + '.fs')
    if (os.path.exists(f_src) and os.path.exists(f_tgt)):
        print(name)
        source = []
        for sent in open(f_src):
            source.append(sent.split())
        tgt = []
        for sent in open(f_tgt):
            tgt.append(sent.split())
        return {'src': source, 'tgt': tgt}
    return None


# Generates and returns the dataframe with the relevant text features
# Loads all the articles in the directory into the dataframe
def format_tv2(args):
    logger.info('Processing tv2')
    # Creates a dataframe with all of the articles in the target directory
    filepaths = [f for f in listdir(args.raw_path) if f.endswith('.json')]
    df_cleaned = pd.concat(map(load_data_tv2, args, filepaths), ignore_index=True)
    print(df_cleaned)
    print('DATA LOADED')

    df = df_cleaned[['body', 'summary']]
    df.columns = ['text', 'summary']
    if (args.type == 'combined'):
        return df_cleaned


    train, validate, test = train_validate_test_split(df, 0.90, 0.05, 242)
    toTxtFile(train, "train","tv2", args)
    toTxtFile(validate, "valid","tv2", args)
    toTxtFile(test, "test", "tv2",args)


# Cleans the given string from remaining html fragments and unwanted characters
def text_cleaner(args, text):
    # Regex to remove remaining HTML fragments from the string
    # and further cleaning of the strings
    s_html = re.sub(r'<.+?>', '', text)
    s = s_html.replace('\n', ' ')
    s = re.sub(' +', ' ', s)
    s = s.replace('\xa0', ' ')
    s = s.replace('´', '\'')
    s = s.replace('`', '\'')
    if (args.botxo == False):
        s = s.replace('å', 'aa')
        s = s.replace('Å', 'aa')
    return s


# Loads the data from the json files into a dataframe and cleans the HTML fragments from the text body
def load_data_tv2(args, path):
    source = args.raw_path + path

    # Retrieve the designated json formatted file from the directory and place in a dataframe
    df = pd.read_json(source, lines=True, orient='columns')
    df = df[['summary', 'body']]

    # List of the different html segments of the article body
    dicts = df['body'].values[0]

    # List of the different paragraphs
    htmls = [item.get('html', "") for item in dicts]

    # Concatenates all the strings
    text = ''.join(htmls)

    # Removes unwanted html fragments from the text
    df['body'] = text_cleaner(args, text)
    df['summary'] = text_cleaner(args, df['summary'].values[0])
    return df

#Splits the dataframe into train, validate and test sets
def train_validate_test_split(df, train_percent=.90, validate_percent=.05, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

#Writes each document to a txt file including a "mapping" file that holds information on what train/valid/test set the document belongs
def toTxtFile(df, mode, datasetName, args):
    print("TO TXT")
    i = 0
    global count
    path = args.raw_path + datasetName + "/mapping/"
    Path(path).mkdir(parents=True, exist_ok=True)
    mapping_file = open(path + "mapping_" + mode + ".txt", 'w', encoding='utf-8')
    for index, row in df.iterrows():
        if i > len(df):
            print("Done")
            break
        else:
            path = args.raw_path + datasetName + "/data/"
            Path(path).mkdir(parents=True, exist_ok=True)
            f = open(path + str(count) + ".txt", 'w', encoding='utf-8')
            body = row['text']
            newBody = str(body + '\n' + '\n' + '@highlight' + '\n' + '\n' + row['summary'])
            f.write(newBody)
            mapping_file.write(str(count) + '\n')
            f.close()
            i += 1
    mapping_file.close()

#Finds all documents of the specific type  either "extractive", "mixed" or "abstrative"
# in the dataframes and returns a dataframe only containing that type.
def type_split(df, type):
    grouped = df.groupby(df.density_bin)
    df = grouped.get_group(type)
    df.index = range(len(df))
    return df

#Formats danewsroom depended on the type you want. Current types = ext,abs,mix,full and combined
def format_danewsroom(args):
    df = pd.read_json(args.zip_path, lines=True)
    if (args.type == "ext"):
        df = type_split(df, "extractive")
        df = clean_danewsroom(args, df)
        exttrain, extvalidate, exttest = train_validate_test_split(df, 0.9, 0.05, 242)
        toTxtFile(exttrain, "train", "extractive", args)
        toTxtFile(extvalidate, "valid", "extractive", args)
        toTxtFile(exttest, "test", "extractive", args)
        logger.info("Done processing extractive")
        logger.info("Done")

    elif (args.type == "abs"):
        df = type_split(df, "abstractive")
        df = clean_danewsroom(args, df)
        abtrain, abvalidate, abtest = train_validate_test_split(df, 0.9, 0.05, 242)
        toTxtFile(abtrain, "train", "abstractive", args)
        toTxtFile(abvalidate, "valid", "abstractive", args)
        toTxtFile(abtest, "test", "abstractive", args)
        logger.info("Done processing abstractive")
        logger.info("Done")

    elif (args.type == "mix"):
        df = type_split(df, "mixed")
        df = clean_danewsroom(args, df)
        mixtrain, mixvalidate, mixtest = train_validate_test_split(df, 0.9, 0.05, 242)
        toTxtFile(mixtrain, "train", "mixed", args)
        toTxtFile(mixvalidate, "valid", "mixed", args)
        toTxtFile(mixtest, "test", "mixed", args)
        logger.info("Done processing mixed")
        logger.info("Done")

    elif (args.type == 'all'):
        df = clean_danewsroom(args, df)
        ext = type_split(df, "extractive")
        ab = type_split(df, "abstractive")
        mix = type_split(df, "mixed")

        exttrain, extvalidate, exttest = train_validate_test_split(ext, 0.9, 0.05, 242)
        mixtrain, mixvalidate, mixtest = train_validate_test_split(mix, 0.9, 0.05, 242)
        abtrain, abvalidate, abtest = train_validate_test_split(ab, 0.9, 0.05, 242)

        toTxtFile(exttrain, "train", "extractive", args)
        toTxtFile(extvalidate, "valid", "extractive", args)
        toTxtFile(exttest, "test", "extractive", args)
        logger.info("Done processing extractive")

        global count
        count = 0
        toTxtFile(mixtrain, "train", "mixed", args)
        toTxtFile(mixvalidate, "valid", "mixed", args)
        toTxtFile(mixtest, "test", "mixed", args)
        logger.info("Done processing mixed", args)

        global count
        count = 0
        toTxtFile(abtrain, "train", "abstractive", args)
        toTxtFile(abvalidate, "valid", "abstractive", args)
        toTxtFile(abtest, "test", "abstractive", args)
        logger.info("Done processing abstractive")
        logger.info("Done")

    elif (args.type == 'full'):
        df = clean_danewsroom(args, df)
        train, validate, test = train_validate_test_split(df, 0.90, 0.05, 242)
        toTxtFile(train, "train", "full", args)
        toTxtFile(validate, "valid", "full", args)
        toTxtFile(test, "test", "full", args)
        logger.info("Done processing full")
        logger.info("Done")
    elif (args.type == 'combined'):
        df = clean_danewsroom(args, df)
        df2 = format_tv2(args)
        df3 = df2.append(df)
        train, validate, test = train_validate_test_split(df3, 0.90, 0.05, 242)
        toTxtFile(train, "train", "combined", args)
        toTxtFile(validate, "valid", "combined", args)
        toTxtFile(test, "test", "combined", args)
        logger.info("Done processing combined")
        logger.info("Done")

#Cleans the text in a dataframe from unwanted tokens.
def clean_danewsroom(args, df):
    for index, row in df.iterrows():
        src = row['text']
        summary = row['summary']
        summary = summary.replace('`', '\'')
        summary = summary.replace('´', '\'')
        if args.botxo == False:
            summary = summary.replace('å', 'aa')
            summary = summary.replace('Å', 'aa')
            src = src.replace('å', 'aa')
            src = src.replace('Å', 'aa')
        src = src.replace('`', '\'')
        src = src.replace('´', '\'')
        df.at[index, 'summary'] = summary
        df.at[index, 'text'] = src
    return df

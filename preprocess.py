import argparse
from collections import defaultdict

import string
import dill
import numpy as np
import pandas as pd
import scipy as sp
import torch
from tqdm import tqdm
from transformers import XLNetTokenizer, XLNetModel, RobertaModel
from rake_nltk import Rake
import yake  # pip3 install git+https://github.com/LIAAD/yake

from GPT_GNN.data import Graph
from GPT_GNN.utils import normalize

parser = argparse.ArgumentParser(description='Preprocess OAG Data')

'''
    Dataset arguments
'''
parser.add_argument('--input_dir', type=str, default='preprocessed/oag_raw',
                    help='The address to store the original data directory.')
parser.add_argument('--output_dir', type=str, default='preprocess_output',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--domain', type=str, default='_CS')
parser.add_argument('--citation_bar', type=int, default=10,
                    help='Only consider papers with citation larger than (2020 - year) * citation_bar')
parser.add_argument('--test_year', type=int, default=2017,
                    help='Papers published after the specific year will be put in the fine-tuning testing dateset.')
args = parser.parse_args()

device = torch.device("cpu")  # Only "cpu" for my computer

cite_dict = defaultdict(lambda: 0)  # Default value for each key is 0
with open(args.input_dir + '/PR%s_20190919.tsv' % args.domain) as fin:  # Use "tsv" file as INPUT
    fin.readline()  # For title
    for l in tqdm(fin, total=sum(1 for line in open(
            args.input_dir + '/PR%s_20190919.tsv' % args.domain))):  # l = ['2001168787', '1963479517']
        l = l[:-1].split('\t')  # Split each element
        cite_dict[l[1]] += 1

pfl = defaultdict(lambda: {})
with open(args.input_dir + '/Papers%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total=sum(1 for line in open(args.input_dir + '/Papers%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        bound = min(2020 - int(l[1]), 20) * args.citation_bar  # USED TO control the size of data in use (based on the diff of published & current years)
        # ReferenceId for the corresponding PaperId must not smaller than the "bound"
        # No empty value for PaperId, PublishYear, NormalisedTitle, VenueId, DetectedLanguage
        # Published Year is no early than 2000 (USED TO control the size of data in use)
        if cite_dict[l[0]] < bound or l[0] == '' or l[1] == '' or l[2] == '' or l[3] == '' and l[4] == '' or int(
                l[1]) < 2000:
            continue
        pi = {'id': l[0], 'title': l[2], 'type': 'paper', 'time': int(l[1])}  # Store column information
        pfl[l[0]] = pi

del cite_dict

# XLNet: Using an auto-regressive method to learn bidirectional contexts by maximizing the expected likelihood
# over all permutations of the input sequence factorization order
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased', output_hidden_states=True, output_attentions=True).to(device)
# Other NLP models to handle abstract differently
roberta_model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True, output_attentions=True).to(device)

# Customize punctuation check list for text data cleaning
punc = string.punctuation + "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
# Key extraction techniques:
rake_nltk_var = Rake()
keywords_num = 20  # The maximum number of keywords from abstract
language = "en"
max_ngram_size = 5  # Limit the maximum words number in an extracted keywords n-gram
deduplication_threshold = 0.9  # Repeat the same words in different key phrases (0.1-duplication, 0.9-NO duplication)
# deduplication_algo = 'seqm'  # Deduplication function [leve|jaro|seqm]
yake = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=keywords_num, features=None)

pfl_emb = defaultdict(lambda: {})
with open(args.input_dir + '/PAb%s_20190919.tsv' % args.domain, errors='ignore') as fin:
    fin.readline()
    for l in tqdm(fin, total=sum(
            1 for line in open(args.input_dir + '/PAb%s_20190919.tsv' % args.domain, 'r', errors='ignore'))):
        try:
            l = l.split('\t')
            if l[0] in pfl:
                abs = l[1]  # Grab string of raw abstract
                abs = abs.lower()  # Convert text to lowercase
                abs = abs.translate(str.maketrans('', '', punc))  # Remove punctuation from the string

                # Keyword extraction for abstract:
                # print("Abstract: \n", abs)
                # RAKE (Rapid Automatic Keyword Extraction algorithm):
                # rake_nltk_var.extract_keywords_from_text(abs)
                # abs_keywords = rake_nltk_var.get_ranked_phrases()
                # # if len(abs_keywords) > keywords_num:
                # #     abs_keywords = abs_keywords[:keywords_num]  # Limit the maximum num of keywords from abstract
                # abs = ' '.join(abs_keywords)
                # YAKE (Yet Another Keyword Extractor):
                abs_keywords = yake.extract_keywords(abs)
                # print(abs_keywords)
                abs = ''
                for kw in abs_keywords:
                    abs = abs + kw[0] + ' '  # Link all keywords together (kw[1] is score: lower -> more relevant)
                abs = abs[:-1]  # Remove the final space
                # print("Final Abstract: \n", abs)

                # Consider abstract embedding:
                input_ids = torch.tensor([tokenizer.encode(pfl[l[0]]['title'])]).to(device)[:, :64]
                abs_input_ids = torch.tensor([tokenizer.encode(abs)]).to(device)[:, :64]  # ADJUST the TOKENIZER for abstract contents
                if len(input_ids[0]) < 4 or len(abs_input_ids[0]) < 4:
                    continue
                all_hidden_states, all_attentions = model(input_ids)[-2:]
                rep = (all_hidden_states[-2][0] * all_attentions[-2][0].mean(dim=0).mean(dim=0).view(-1, 1)).sum(dim=0)
                abs_all_hidden_states, abs_all_attentions = roberta_model(abs_input_ids)[-2:]  # ADJUST the MODEL for abstract contents
                abs_rep = (abs_all_hidden_states[-2][0] * abs_all_attentions[-2][0].mean(dim=0).mean(dim=0).view(-1, 1)).sum(dim=0)
                pfl_emb[l[0]] = pfl[l[0]]
                pfl_emb[l[0]]['emb'] = rep.tolist()  # pfl_emb will not involve any paper without 'emb'
                pfl_emb[l[0]]['abs_emb'] = abs_rep.tolist()  # Add abstract embedding to the dictionary

                # # Only consider title embedding:
                # input_ids = torch.tensor([tokenizer.encode("na")]).to(device)[:, :64]  # Specially test for empty-content string title
                # # input_ids = torch.tensor([tokenizer.encode(pfl[l[0]]['title'])]).to(device)[:, :64]  # With title contents
                # if len(input_ids[0]) < 4:
                #     continue
                # all_hidden_states, all_attentions = model(input_ids)[-2:]
                # rep = (all_hidden_states[-2][0] * all_attentions[-2][0].mean(dim=0).mean(dim=0).view(-1, 1)).sum(dim=0)
                # pfl_emb[l[0]] = pfl[l[0]]
                # pfl_emb[l[0]]['emb'] = rep.tolist()

                # # Consider title and abstract in one embedding:
                # input_ids = torch.tensor([tokenizer.encode(pfl[l[0]]['title'] + abs)]).to(device)[:, :64]
                # if len(input_ids[0]) < 4:
                #     continue
                # all_hidden_states, all_attentions = model(input_ids)[-2:]
                # rep = (all_hidden_states[-2][0] * all_attentions[-2][0].mean(dim=0).mean(dim=0).view(-1, 1)).sum(dim=0)
                # pfl_emb[l[0]] = pfl[l[0]]
                # pfl_emb[l[0]]['emb'] = rep.tolist()
        except Exception as e:
            print(e)
del pfl

vfi_ids = {}
with open(args.input_dir + '/vfi_vector.tsv') as fin:
    for l in tqdm(fin, total=sum(1 for line in open(args.input_dir + '/vfi_vector.tsv'))):
        l = l[:-1].split('\t')  # Ignore the last element in the list
        vfi_ids[l[0]] = True  # Add the 'True' value to the corresponding key - 1st element in the line

graph = Graph()

rem = []
with open(args.input_dir + '/Papers%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total=sum(1 for line in open(args.input_dir + '/Papers%s_20190919.tsv' % args.domain, 'r'))):
        l = l[:-1].split('\t')
        if l[0] not in pfl_emb or l[4] != 'en' or l[3] not in vfi_ids:
            continue
        rem += [l[0]]
        vi = {'id': l[3], 'type': 'venue', 'attr': l[-2]}
        graph.add_edge(pfl_emb[l[0]], vi, time=int(l[1]), relation_type='PV_' + l[-2])
del rem

with open(args.input_dir + '/PR%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total=sum(1 for line in open(args.input_dir + '/PR%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        if l[0] in pfl_emb and l[1] in pfl_emb:
            p1 = pfl_emb[l[0]]
            p2 = pfl_emb[l[1]]
            if p1['time'] >= p2['time']:
            # if p1['time'] >= p2['time'] and p1['time'] <= args.test_year:  # Break testing links
                graph.add_edge(p1, p2, time=p1['time'], relation_type='PP_cite')

ffl = {}
with open(args.input_dir + '/PF%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total=sum(1 for line in open(args.input_dir + '/PF%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        if l[0] in pfl_emb and l[1] in vfi_ids:
            ffl[l[1]] = True

with open(args.input_dir + '/FHierarchy_20190919.tsv') as fin:
    fin.readline()
    for l in tqdm(fin, total=sum(1 for line in open(args.input_dir + '/FHierarchy_20190919.tsv'))):
        l = l[:-1].split('\t')
        if l[0] in ffl and l[1] in ffl and l[0] in pfl_emb:
        # if l[0] in ffl and l[1] in ffl and l[0] in pfl_emb \
                # and pfl_emb[l[0]]['time'] <= args.test_year and pfl_emb[l[1]]['time'] <= args.test_year:  # Break testing links
            fi = {'id': l[0], 'type': 'field', 'attr': l[2]}
            fj = {'id': l[1], 'type': 'field', 'attr': l[3]}
            graph.add_edge(fi, fj, relation_type='FF_in')
            ffl[l[0]] = fi
            ffl[l[1]] = fj

with open(args.input_dir + '/PF%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total=sum(1 for line in open(args.input_dir + '/PF%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        if l[0] in pfl_emb and l[1] in ffl and type(ffl[l[1]]) == dict:
        # if l[0] in pfl_emb and l[1] in ffl and type(ffl[l[1]]) == dict \
        #         and pfl_emb[l[0]]['time'] <= args.test_year:  # Break testing links
            pi = pfl_emb[l[0]]
            fi = ffl[l[1]]
            graph.add_edge(pi, fi, time=pi['time'], relation_type='PF_in_' + fi['attr'])

del ffl

coa = defaultdict(lambda: {})
with open(args.input_dir + '/PAuAf%s_20190919.tsv' % args.domain) as fin:
    fin.readline()
    for l in tqdm(fin, total=sum(1 for line in open(args.input_dir + '/PAuAf%s_20190919.tsv' % args.domain))):
        l = l[:-1].split('\t')
        if l[0] in pfl_emb and l[2] in vfi_ids:
            pi = pfl_emb[l[0]]
            ai = {'id': l[1], 'type': 'author'}
            fi = {'id': l[2], 'type': 'affiliation'}
            coa[l[0]][int(l[-1])] = ai
            graph.add_edge(ai, fi, relation_type='in')

del vfi_ids

for pid in tqdm(coa):
    if pid not in pfl_emb:
        continue
    pi = pfl_emb[pid]
    max_seq = max(coa[pid].keys())
    for seq_i in coa[pid]:
        ai = coa[pid][seq_i]
        # if pi['time'] <= args.test_year:  # Break testing links
        if seq_i == 1:
            graph.add_edge(ai, pi, time=pi['time'], relation_type='AP_write_first')
        elif seq_i == max_seq:
            graph.add_edge(ai, pi, time=pi['time'], relation_type='AP_write_last')
        else:
            graph.add_edge(ai, pi, time=pi['time'], relation_type='AP_write_other')

del coa

with open(args.input_dir + '/vfi_vector.tsv') as fin:
    for l in tqdm(fin, total=sum(1 for line in open(args.input_dir + '/vfi_vector.tsv'))):
        l = l[:-1].split('\t')
        ser = l[0]
        for idx in ['venue', 'field', 'affiliation']:
            if ser in graph.node_forward[idx] and ser in pfl_emb:  # idx is the node name, ser is the node id
                graph.node_bacward[idx][graph.node_forward[idx][ser]]['node_emb'] = np.array(l[1].split(' '))

with open(args.input_dir + '/SeqName%s_20190919.tsv' % args.domain, errors='ignore') as fin:
    for l in tqdm(fin, total=sum(
            1 for line in open(args.input_dir + '/SeqName%s_20190919.tsv' % args.domain, errors='ignore'))):
        l = l[:-1].split('\t')
        key = l[2]
        if key in ['conference', 'journal', 'repository', 'patent']:
            key = 'venue'
        if key == 'fos':
            key = 'field'
        if l[0] in graph.node_forward[key]:
            s = graph.node_bacward[key][graph.node_forward[key][l[0]]]
            s['name'] = l[1]

'''
    Calculate the total citation information as node attributes.
'''
for idx, pi in enumerate(graph.node_bacward['paper']):
    pi['citation'] = len(graph.edge_list['paper']['paper']['PP_cite'][idx])
for idx, ai in enumerate(graph.node_bacward['author']):
    citation = 0
    for rel in graph.edge_list['author']['paper'].keys():
        for pid in graph.edge_list['author']['paper'][rel][idx]:
            citation += graph.node_bacward['paper'][pid]['citation']
    ai['citation'] = citation
for idx, fi in enumerate(graph.node_bacward['affiliation']):
    citation = 0
    for aid in graph.edge_list['affiliation']['author']['in'][idx]:
        citation += graph.node_bacward['author'][aid]['citation']
    fi['citation'] = citation
for idx, vi in enumerate(graph.node_bacward['venue']):
    citation = 0
    for rel in graph.edge_list['venue']['paper'].keys():
        for pid in graph.edge_list['venue']['paper'][rel][idx]:
            citation += graph.node_bacward['paper'][pid]['citation']
    vi['citation'] = citation
for idx, fi in enumerate(graph.node_bacward['field']):
    citation = 0
    for rel in graph.edge_list['field']['paper'].keys():
        for pid in graph.edge_list['field']['paper'][rel][idx]:
            citation += graph.node_bacward['paper'][pid]['citation']
    fi['citation'] = citation

'''
    Since only paper have w2v embedding, we simply propagate its
    feature to other nodes by averaging neighborhoods.
    Then, we construct the Dataframe for each node type.
'''
d = pd.DataFrame(graph.node_bacward['paper'])
graph.node_feature = {'paper': d}
cv = np.array((list(d['emb'])))
abs_cv = np.array((list(d['abs_emb'])))
# print("cv shape:", cv.shape)
# print("cv type:", type(cv))
# print("abs shape:", abs_cv.shape)

test_time_bar = 2016  # Specially designed for "time as classification"
for _type in graph.node_bacward:
    if _type not in ['paper', 'affiliation']:
        d = pd.DataFrame(graph.node_bacward[_type])
        i = []
        for _rel in graph.edge_list[_type]['paper']:
            for t in graph.edge_list[_type]['paper'][_rel]:
                for s in graph.edge_list[_type]['paper'][_rel][t]:
                    if graph.edge_list[_type]['paper'][_rel][t][s] <= test_time_bar:
                        i += [[t, s]]
        if len(i) == 0:
            continue
        i = np.array(i).T
        v = np.ones(i.shape[1])
        m = normalize(
            sp.sparse.coo_matrix((v, i), shape=(len(graph.node_bacward[_type]), len(graph.node_bacward['paper']))))
        del i
        del v
        m = m.toarray()  # I added
        # print("m shape:", m.shape)
        # print("m successful!")

        d['emb'] = list(m.dot(cv))
        # print("d-emb successful!")
        d['abs_emb'] = list(m.dot(abs_cv))
        # print("d-abs_emb successful!")
        del m
        graph.node_feature[_type] = d
        # print("graph-node-f successful!")
        del d
del cv
del abs_cv
del test_time_bar
# print("successful!")

'''
    Affiliation is not directly linked with Paper, so we average the author embedding.
'''
cv = np.array(list(graph.node_feature['author']['emb']))
# print("cv shape:", cv.shape)
# print("cv type:", type(cv)
d = pd.DataFrame(graph.node_bacward['affiliation'])
i = []
for _rel in graph.edge_list['affiliation']['author']:
    for j in graph.edge_list['affiliation']['author'][_rel]:
        for t in graph.edge_list['affiliation']['author'][_rel][j]:
            i += [[j, t]]
i = np.array(i).T
v = np.ones(i.shape[1])
m = normalize(
    sp.sparse.coo_matrix((v, i), shape=(len(graph.node_bacward['affiliation']), len(graph.node_bacward['author']))))
del i
del v
m = m.toarray()  # I added
# print("m shape:", m.shape)
# print("m successful!")
d['emb'] = list(m.dot(cv))
del m
del cv
# print("d-emb successful!")
graph.node_feature['affiliation'] = d
del d
# print("successful!")
del pfl_emb

edg = {}
for k1 in graph.edge_list:
    if k1 not in edg:
        edg[k1] = {}
    for k2 in graph.edge_list[k1]:
        if k2 not in edg[k1]:
            edg[k1][k2] = {}
        for k3 in graph.edge_list[k1][k2]:
            if k3 not in edg[k1][k2]:
                edg[k1][k2][k3] = {}
            for e1 in graph.edge_list[k1][k2][k3]:
                if len(graph.edge_list[k1][k2][k3][e1]) == 0:
                    continue
                edg[k1][k2][k3][e1] = {}
                for e2 in graph.edge_list[k1][k2][k3][e1]:
                    edg[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]
graph.edge_list = edg

del edg
del graph.node_bacward

dill.dump(graph, open(args.output_dir + '/graph_5gram_key20_yake_x.pk', 'wb'))
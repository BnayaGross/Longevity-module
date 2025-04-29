#!/usr/bin/env python
# coding: utf-8



import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import glob
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.interpolate import splev
from scipy.interpolate import splrep
import colorsys
import networkx as nx
import random
import math
import matplotlib.style
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import multiprocessing as mp
import json
import sys
from scipy.stats import hypergeom
from tqdm import tqdm
import networkx.algorithms.community as nx_comm
import disease_module_identification as dmi
import NetworkMetrics as metrics
from scipy import stats
import proximity.distances
import proximity.network
import multiprocessing



def prox_adapter(G_sub,target_genes,module_longevity,D, drug, am):
    if len(target_genes) == 0:
        new_row = {'drug': drug, 'number_of_target_genes': len(target_genes), 'prox': 'nan' ,'p-value': 'nan', 'z_score': 'nan', 'aging_mechanisms_group': am}
    else:
        prox = metrics.proximity(G_sub,target_genes,module_longevity,D,degree_preserving = 'log_binning')
        new_row = {'drug': drug, 'number_of_target_genes': len(target_genes), 'prox': prox['raw_amspl'], 'p-value': prox['p_val'], 'z_score': prox['z_score'], 'aging_mechanisms_group': am}
      
    return new_row


# load PPI_2022

path_interactome = "./data/PPI_2022.csv"
G = nx.from_pandas_edgelist(pd.read_csv(path_interactome), 'HGNC_Symbol.1', 'HGNC_Symbol.2')
print("PPI 2022 loaded")

self_loops = [(u, v) for u, v in G.edges() if u == v]
G.remove_edges_from(self_loops)

connected_components = list(nx.connected_components(G))
lcc = max(len(component) for component in connected_components)


path = "./data/Gene_hallmarks.csv"
gene_hallmarks_extended = pd.read_csv(path)
gene_hallmarks_extended = gene_hallmarks_extended[['GeneId','aging_mechanisms','criteria','confidence','aging_mechanisms_group' ]]

print("Hallmark genes loaded")


path_all_drugbank_drugs = "./data/demo_drugs.csv"
all_drugbank_drugs = pd.read_csv(path_all_drugbank_drugs)
approved_drugs_names = all_drugbank_drugs['Name'].unique()


print("DrugBank loaded")



largest_cc = max(nx.connected_components(G), key=len)
G_sub = G.subgraph(largest_cc)
target_genes_bucket = []
for idx, drug in enumerate(approved_drugs_names):
    target_genes = all_drugbank_drugs[all_drugbank_drugs['Name'] == drug]
    target_genes_bucket.append(set(target_genes[target_genes.Gene_Target.isin(G_sub.nodes())]['Gene_Target']))


print("Bucket of all drug targets created")

filename_dis = "./data/PPI_2022_distances.pkl"
D = metrics.load_distances(filename_dis)

print("Distance matrix loaded")

print("Start measuring proximity")


level = int(sys.argv[1])

A = gene_hallmarks_extended[gene_hallmarks_extended['confidence'] <= level]
amg_list = list(A['aging_mechanisms_group'].unique())
amg_list.remove('other')

compunds_pval = pd.DataFrame()
for am in amg_list:
    longevity_genes = set(A[A['aging_mechanisms_group'] == am]['GeneId']) & set(G_sub.nodes())
    
    for idx in range(len(approved_drugs_names)):
        results = prox_adapter(G_sub,target_genes_bucket[idx],longevity_genes,D,approved_drugs_names[idx],  am)
        compunds_pval = pd.concat([compunds_pval, pd.DataFrame([results])], ignore_index=True)
        print(am, approved_drugs_names[idx])
        
        

compunds_pval.to_csv('./Output/Proximity_drugs_log_binning_level' + str(level) + '.csv', index=False) 


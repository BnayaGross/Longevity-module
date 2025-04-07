The files in this folder are used to obtain the results presented in the article entitled "Network-driven discovery of repurposable drugs targeting hallmarks of aging".

The following folders are presented:
1. The folder "Proximity and pAGE results" contains the results of the algorithms and provides the Proximity and pAGE values for each drug to each hallmark of aging for every confidence level.
2. The folder "proximity" contains functions used by the main scripts.
3. The folder "data" contains the data used by the algorithm to obtain the results. Specifically, it contains the following files:
   3.1 "PPI_2022.csv" - The human interactome.
   3.2 "Gene_hallmarks.csv" - The aging genes and their association with the hallmark of aging. This data is obtained from the OpenGenes database.
   3.3 "all_drugbank_drugs.csv" -  All drugs and their targets. This data is obtained from the DrugBank database.
   3.4 "age-related-changes.tsv" - Aging related expression changes used to calculate pAGE. This data is obtained from the OpenGenes database.
   3.5 "PPI_2022_distances.pkl" - All shortest paths between pairs of nodes in the interactome. This file is very big and cannnot be uploaded to GitHub. Please use the script "Create_PPI_2022_distances.ipynb" to generate it.
4. The folder "CMap_data" contains the expression data of the drugs perturbation used by the algorithm to obtain the results. Specifically, it contains the following files:
   4.1 "compoundinfo_beta.txt" - Compounds metadata information.
   4.2 "geneinfo_beta.txt" - Genes metadata information.
   4.3 "level5_beta_trt_cp_n720216x12328.gctx" - Expression data of drugs perturbation. This file is very big and cannnot be uploaded to GitHub. Please download it from the CMap database - "https://clue.io/data/GCTX#GCTxDatasets" . See also the "README.txt" file in the CMap_data folder.
   4.4 "siginfo_beta.txt" - Signature metadata information.

The following scripts contain functions used by the main scripts:
1. "NetworkMetrics.py" - is a short subset librery of functions from the NetMedPy package - https://github.com/menicgiulia/NetMedPy .
2. "Diamond.py" and "disease_module_identification.py" contains function for calculating network properties.
3. "Create_PPI_2022_distances.ipynb" - Creates the file "PPI_2022_distances.pkl" used to calculate the proximity.

The following scripts are the main scripts:
   







   

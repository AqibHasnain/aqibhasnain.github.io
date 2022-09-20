---
title: 'Metric spaces of microbial essential gene landscapes'
date: 2022-09-25
permalink: /posts/2022/09/metric-learning-essential-genes/
tags:
  - metric learning
  - deep learning
  - manifold learning
  - dimensionality reduction
  - essential genes
  - microbes
---

Essential genes are those which are crucial for survival of an organism in a given context. This post will introduce manifold and metric learning to characterize and classify essential genes from the chaos game representation of a genetic sequence.

Briefly, what are metric spaces? 
======

Can we learn a metric on a space from data points? 
======

Learning essential gene landscapes
======

The Database of Essential Genes
------
The [Database of Essential Genes](http://origin.tubic.org/deg/public/index.php) is a repository of the essential genes from roughly 100 organisms spanning bacteria, archaea, and eukaryotes. Under the [download](http://origin.tubic.org/deg/public/index.php/download) tab you will find links to download gene sequences as nucleic acids or as amino acids along with their corresponding annotations e.g. gene identifiers, media conditions of the experiments, etc. 

In this post, we will analyze the essential genes of bacteria, so let's start by reading that annotation csv file `deg_annotation_p.csv` into a python notebook and prepare it a bit: 

```python 
anno_df = pd.read_csv('/Users/aqib/downloads/deg_annotation_p.csv', sep=';', header=None,quotechar='"')

# the columns need to be named (some cols couldn't be understood by context)
anno_df.columns = ['organism_id','gene_id','gene','protein_id?','?','biological process','biological function','bacteria','genome_ref','media','locus_tag','GO_id','??','???']

```

Parsing the database for essential gene sequences
------

Grabbing non-essential gene sequences from NCBI
------

Embedding gene sequences in vector spaces -- Chaos Game Representation
------

What can we learn from unsupervised learning approaches? 
------

Metric learning with the triplet margin loss implemented in Pytorch
------

Classification in the embedded space
------
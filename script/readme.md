# A guidance for script

Here is the base script containing three terms. 

```
bash script/run.sh DATASET BACKBONE SAMPLING
```

The first is `dataset`, which indicates the source and target of the transfer learning. We provide a specific guide on how to set the `dataset` in the following.

The second is `backbone`, which can be chosen from `gcn`, `gat`, `sage`, `sgc`. More architectures can be implemented under `model.py`. 

The last term is `SAMPLING` involving the subgraph sampling method in our proposed SP. we provide two `sampling` functions including `k_hop` (k-hop sampler) and `rw` (random walk sampler), corresponding to SP and SP++, respectively. 

Here are some examples. 

```
bash script/run.sh acm_dblp gcn k_hop
bash script/run.sh dblp_acm gcn k_hop
bash script/run.sh arxiv_1_arxiv_5 gcn rw
bash script/run.sh arxiv_3_arxiv_5 gcn rw
```

## How to set `Dataset` in script? 

**Citation Network**

There are two datasets: ACM and DBLP. We test the one-to-one transfer setting.  

- Transfer From ACM to DBLP: `acm_dblp`
- Transfer From DBLP to ACM: `dblp_acm`

**Airport Network**

There are three datasets: USA, Europe, Brazil. We test the one-to-one transfer setting. 

- Transfer From USA to Europe: `usa_europe`
- Transfer From USA to Brazil: `usa_brazil`
- Transfer From Europe to USA: `europe_usa`
- Transfer From Europe to Brazil: `europe_brazil`
- Transfer From Brazil to USA: `brazil_usa`
- Transfer From Brazil to Europe: `brazil_europe`

**Twitch Network**

There are six networks collected from different countries. We test the one-to-multi transfer learning performance. Particularly, the knowledge is transferred from DE to the remaining datasets, i.e., EN, ES, FR, PT, RU. 

- Transfer From DE to EN: `de_en`
- Transfer From DE to ES: `de_es`
- Transfer From DE to FR: `de_fr`
- Transfer From DE to PT: `de_pt`
- Transfer From DE to RU: `de_ru`

**Arxiv Network**

We test the temporal dynamic distribution shift on the dataset. Specifically, this is a citation network where papers are published from 2005 to 2020. We consider five splits: *Time 1* (2005 - 2007), *Time 2* (2008 - 2010), *Time 3* (2011 - 2014), *Time 4* (2015 - 2017), *Time 5* (2018 - 2020). We transfer the knowledge from the previous four datasets to the last one. 

- Transfer From T1 to T5: `arxiv_1_arxiv_5`
- Transfer From T2 to T5: `arxiv_2_arxiv_5`
- Transfer From T3 to T5: `arxiv_3_arxiv_5`
- Transfer From T4 to T5: `arxiv_4_arxiv_5`

Additionally, we also consider one domain adaptation setting (Degree). 

- `arxiv_arxiv_0`

**Elliptic Network**

This is another temporal dynamic graph. Please directly run the specific script. 

```
bash script/run_elliptic.sh elliptic BACKBONE SAMPLING
```

The example is 

```
bash script/run_elliptic.sh elliptic gat rw
```

**Facebook Network**

This is a social dataset collected on Facebook, which consists of 14 networks. Please run the specific script. 

```
bash script/run_fb.sh DATASET BACKBONE SAMPLING
```

The model is trained on 3 graphs, validated on 2 graphs, and test on 1 graph. Here is an example

```
bash script/run_fb.sh facebook_1_2_3_10 gcn k_hop
```

The `dataset` looks like `facebook_1_2_3_10` where different numbers indicate different graphs. We use the graphs with first three indices as training graphs and the last one as testing. In this example, `1, 2, 3` are training graphs and `10` is testing graph. Note that graphs with indices `13` and `14` are used as validations for all of settings. 

Here is the specific mapping: 
1. Johns Hopkins55
2. Caltech36
3. Amherst41
4. Bingham82
5. Duke14
6. Princeton12
7. WashU32
8. Brandeis99
9. Carnegie49
10. Penn94
11. Brown11
12. Texas80
13. Cornell5
14. Yale4
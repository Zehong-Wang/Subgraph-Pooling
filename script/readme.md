# How to set source_target in params? 

**Citation Network**

There are two datasets: ACM and DBLP. We test the one-to-one transfer setting.  

- Transfer From ACM to DBLP: acm_dblp
- Transfer From DBLP to ACM: dblp_acm

**Airport Network**

There are three datasets: USA, Europe, Brazil. We test the one-to-one transfer setting. 

- Transfer From USA to Europe: usa_europe
- Transfer From USA to Brazil: usa_brazil
- Transfer From Europe to USA: europe_usa
- Transfer From Europe to Brazil: europe_brazil
- Transfer From Brazil to USA: brazil_usa
- Transfer From Brazil to Europe: brazil_europe

**Twitch Network**

There are six networks collected from different countries. We test the one-to-multi transfer learning performance. Particularly, the knowledge is transferred from DE to the remaining datasets, i.e., EN, ES, FR, PT, RU. 

- Transfer From DE to EN: de_en
- Transfer From DE to ES: de_es
- Transfer From DE to FR: de_fr
- Transfer From DE to PT: de_pt
- Transfer From DE to RU: de_ru

**Arxiv Network**

We test the temporal dynamic distribution shift on the dataset. Specifically, this is a citation network where papers are published from 2005 to 2020. We consider five splits: *Time 1* (2005 - 2007), *Time 2* (2008 - 2010), *Time 3* (2011 - 2014), *Time 4* (2015 - 2017), *Time 5* (2018 - 2020). We transfer the knowledge from the previous four datasets to the last one. 

- Transfer From T1 to T5: arxiv_1_arxiv_5
- Transfer From T2 to T5: arxiv_2_arxiv_5
- Transfer From T3 to T5: arxiv_3_arxiv_5
- Transfer From T4 to T5: arxiv_4_arxiv_5

Additionally, we also consider one domain adaptation setting (Degree). 

- arxiv_arxiv_0

**Elliptic Network**

This is another temporal dynamic graph. Please directly run the specific code. 

**Facebook Network**

This is a social network collected on Facebook. Please run the specific code. 
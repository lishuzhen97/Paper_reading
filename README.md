# 论文阅读
---
NLP相关的论文，尤其是**文本自动摘要**相关的文章和代码

- [深度学习NLP](#深度学习nlp)
- [预训练模型](#预训练模型)
- [摘要抽取](#摘要抽取)
- [摘要生成](#摘要生成)
- [谣言检测](#谣言检测陈述提取)
- [情感分析](#情感分析)


## 深度学习NLP
- **LSTM**:"Long Short-Term Memory".Neural Comput(1997)[[PDF]](https://doi.org/10.1162/neco.1997.9.8.1735)
- **TextRank**: "TextRank: Bringing Order into Text". ACL(2004)[[PDF]](https://aclanthology.org/W04-3252/)
- **Sequence to Sequence**: "Sequence to Sequence Learning with Neural Networks".  NIPS(2014) [[PDF]](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)
- **Transformer**: "Attention is All you Need". NeurIPS(2017) [[PDF]](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)[[MyNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/attention-is-all-you-need-Paper.pdf)
- **New Log-linear**: "Efficient Estimation of Word Representations in Vector Space" ICLR(2013) [[PDF]](http://arxiv.org/abs/1301.3781)
- **Transformer-XL**: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1285) [[code]](https://github.com/kimiyoung/transformer-xl)
[[code-official]](https://github.com/tensorflow/tensor2tensor) [[code-tf]](https://github.com/Kyubyong/transformer) [[code-py]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/transform_XL.pdf)
- **Pointer Networks** "Pointer Networks". NIPS(2015) [[PDF]](https://proceedings.neurips.cc/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf)
## 预训练模型
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1423) [[code]](https://github.com/google-research/bert) [[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/BERT.pdf) :star::star::star::star::star:
-  **ALBERT**: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations". ICLR(2020) [[PDF]](https://openreview.net/pdf?id=H1eA7AEtvS)
-  **RoBERTa**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1907.11692.pdf) [[code]](https://github.com/pytorch/fairseq)
-  **BIG BiRD**: "Big Bird: Transformers for Longer Sequences". NeurIPS(2020)[[PDF]](https://proceedings.neurips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)
-  
## 摘要抽取
- **HIBERT**: "HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization". ACL(2019) [[PDF]](https://doi.org/10.18653/v1/p19-1499)[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/HIBERT_%E6%9C%AA%E5%BC%80%E6%BA%90.pdf)
- **SummaRuNNer**: "SummaRuNNer: A Recurrent Neural Network Based Sequence Model for Extractive Summarization of Documents". ACL(2019) [[PDF]](http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14636) [[code]](https://github.com/hpzhao/SummaRuNNer
)[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/SummaRuNNer.pdf)
-  **SciBERTSUM**: "SciBERTSUM: Extractive Summarization for Scientific Documents". CoRR (2022)[[PDF]](https://arxiv.org/abs/2201.08495)[[code]](https://github.com/atharsefid/SciBERTSUM)[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/SciBERTSUM.pdf)
-  **BERT for Ad Hoc**: "Simple Applications of BERT for Ad Hoc Document Retrieval". CoRR(2019)[[PDF]](http://arxiv.org/abs/1903.10972)[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/Simple%20Applications%20of%20BERT%20for%20Ad%20Hoc%20Document%20Retrieval.pdf)
-  **REFRESH**: "Ranking Sentences for Extractive Summarization
with Reinforcement Learning". NAACL-HLT (2018)[[PDF]](https://doi.org/10.18653/v1/n18-1158)
-  **fastNLP**: "Searching for Effective Neural Extractive Summarization:
WhatWorks and What’s Next". ACL(2019) [[PDF]](https://doi.org/10.18653/v1/p19-1100)
-  **Three Sentences** "Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction". ACL(2021)[[PDF]](https://aclanthology.org/2021.acl-short.126/)[[code]](https://github.com/AndrewZhe/Three-Sentences-Are-All-You-Need)
-  **Sentences and Words** "Neural Summarization by Extracting Sentences and Words". ACL(2016) [[PDF]](https://aclanthology.org/P16-1046/)
-  **SWAP-NET** "Extractive Summarization with SWAP-NET: Sentences andWords from Alternating Pointer Networks". ACL(2018) [[PDF]](https://aclanthology.org/P18-1014/)[[code]](https://github.com/aishj10/swap-net)
-  **Leveraging BERT** "Leveraging BERT for Extractive Text Summarization on
Lectures" CoRR(2019) [[PDF]](http://arxiv.org/abs/1906.04165)[[code]](https://github.com/dmmiller612/lecture-summarizer)[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/Leveraging%20BERT%20for%20Extractive%20Text%20Summarization%20on%20Lectures.pdf)
-  **BERTSum**: "Fine-tune BERT for Extractive Summarization". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1903.10318.pdf) [[code]](https://github.com/nlpyang/BertSum)
-  **NeuSum**: "Neural Document Summarization by Jointly Learning to Score and Select Sentences". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1061) 
## 摘要生成

## 谣言检测/陈述提取
- **EANN**: "EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection". KDD(2018) [[PDF]](https://doi.org/10.1145/3219819.3219903)[[code]](https://github.com/search?q=EANN)[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/EANN_%20Event%20Adversarial%20Neural%20Networks%20for%20Multi-Modal%20Fake%20News%20Detection.pdf)
- **FEVER**: "FEVER: a Large-scale Dataset for Fact Extraction and VERification". NAACL-HLT(2018) [[PDF]](https://doi.org/10.18653/v1/n18-1074)[[code]](https://github.com/awslabs/fever)
- **Claim extraction**: "Claim extraction from text using transfer learning". ICON(2020)[[PDF]](https://aclanthology.org/2020.icon-main.39)[[code]](https://github.com/ashish6630/Claim extraction.git)[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/Claim%20Extraction%20from%20Text%20using%20Transfer%20Learning.pdf)
- **Biomedical Publications**: "Claim Extraction in Biomedical Publications using Deep Discourse Model and Transfer Learning". CoRR(2019) [[PDF]](http://arxiv.org/abs/1907.00962)[[code]](https://github.com/titipata/detecting-scientific-claim)
- **CREDO**: "Neural Network Architecture for Credibility Assessment of Textual Claims". CoRR(2018) [[PDF]](http://arxiv.org/abs/1803.10547) [[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/Neural%20Network%20Architecture%20for%20Credibility%20Assessment%20of%20Textual%20%20Claims.pdf)
- **SVM-based**: "Extracting Important Sentences with Support Vector Machines". COLING(2002) [[PDF]](https://aclanthology.org/C02-1053/)
- **Open Extraction**: "Open Extraction of Fine-Grained Political Statements" EMNLP(2015)
[[PDF]](https://doi.org/10.18653/v1/d15-1008)[[code]](https://people.ischool.berkeley.edu/~dbamman/emnlp2015/)
- **LG + SR**: "Credibility Assessment of Textual Claims on the Web". CIKM(2016) [[PDF]](https://doi.org/10.1145/2983323.2983661) [[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/Credibility%20Assessment%20of%20Textual%20Claims%20on%20the%20Web.pdf)
- **PubMed** "Claim Extraction in Biomedical Publications using Deep Discourse Model and Transfer Learning". CoRR(2019) [[PDF]](http://arxiv.org/abs/1907.00962)[[code]](https://github.com/titipata/detecting-scientific-claim)
- **HoVer** "HOVER: A Dataset for Many-Hop Fact Extraction And Claim Verification". EMNLP(2020)[[code]](https://doi.org/10.18653/v1/2020.findings-emnlp.309)
- **
## 情感分析
- **SC-CMC-KS**: "Sentiment classification model for Chinese micro-blog comments based on key sentences extraction". [[PDF]](https://link.springer.com/content/pdf/10.1007/s00500-020-05160-8.pdf)

- 

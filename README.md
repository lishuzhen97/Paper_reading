# 论文阅读
---
NLP相关的论文，尤其是**文本自动摘要**相关的文章和代码

- [数据集](#数据集)
- [深度学习NLP](#深度学习nlp)
- [预训练模型](#预训练模型)
- [摘要抽取](#摘要抽取)
- [摘要生成](#摘要生成)
- [谣言检测](#谣言检测陈述提取)
- [情感分析](#情感分析)

## 数据集
- **LCSTS**："Global Encoding for Abstractive Summarization".EMNLP(2015)[[PDF]](https://aclanthology.org/D15-1229.pdf)
## 深度学习NLP
- **LSTM**:"Long Short-Term Memory".Neural Comput(1997)[[PDF]](https://doi.org/10.1162/neco.1997.9.8.1735)
- **TextRank**: "TextRank: Bringing Order into Text". ACL(2004)[[PDF]](https://aclanthology.org/W04-3252/)
- **Sequence to Sequence**: "Sequence to Sequence Learning with Neural Networks".  NIPS(2014) [[PDF]](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)
- **Transformer**: "Attention is All you Need". NeurIPS(2017) [[PDF]](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)[[MyNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/attention-is-all-you-need-Paper.pdf)
- **New Log-linear**: "Efficient Estimation of Word Representations in Vector Space" ICLR(2013) [[PDF]](http://arxiv.org/abs/1301.3781)
- **Transformer-XL**: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context". ACL(2019) [[PDF]](https://www.aclweb.org/anthology/P19-1285) [[code]](https://github.com/kimiyoung/transformer-xl)
[[code-official]](https://github.com/tensorflow/tensor2tensor) [[code-tf]](https://github.com/Kyubyong/transformer) [[code-py]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/transform_XL.pdf)
- **Pointer Networks**: "Pointer Networks". NIPS(2015)  [[PDF]](https://proceedings.neurips.cc/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf)
- **GAN**: "Generative Adversarial Nets". NIPS(2014) [[PDF]](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) [[code]](http://www.github.com/goodfeli/adversarial)
## 预训练模型
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL(2019) [[PDF]](https://www.aclweb.org/anthology/N19-1423) [[code]](https://github.com/google-research/bert) [[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/BERT.pdf) :star::star::star::star::star:
- **BoostingBERT**: "BoostingBERT:Integrating Multi-Class Boosting into BERT for NLP". CoRR(2019) [[PDF]](https://arxiv.org/abs/2009.05959)[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/BoostingBERT%20Integrating%20Multi-Class%20Boosting%20into%20BERT%20for%20NLP%20task.pdf)
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
-  **Three Sentences**: "Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction". ACL(2021)[[PDF]](https://aclanthology.org/2021.acl-short.126/)[[code]](https://github.com/AndrewZhe/Three-Sentences-Are-All-You-Need)
-  **Sentences and Words**: "Neural Summarization by Extracting Sentences and Words". ACL(2016) [[PDF]](https://aclanthology.org/P16-1046/)
-  **SWAP-NET** "Extractive Summarization with SWAP-NET: Sentences andWords from Alternating Pointer Networks". ACL(2018) [[PDF]](https://aclanthology.org/P18-1014/)[[code]](https://github.com/aishj10/swap-net)
-  **Leveraging BERT**: "Leveraging BERT for Extractive Text Summarization on
Lectures" CoRR(2019) [[PDF]](http://arxiv.org/abs/1906.04165)[[code]](https://github.com/dmmiller612/lecture-summarizer)[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/Leveraging%20BERT%20for%20Extractive%20Text%20Summarization%20on%20Lectures.pdf)
-  **BERTSum**: "Fine-tune BERT for Extractive Summarization". arXiv(2019) [[PDF]](https://arxiv.org/pdf/1903.10318.pdf) [[code]](https://github.com/nlpyang/BertSum)
-  **NeuSum**: "Neural Document Summarization by Jointly Learning to Score and Select Sentences". ACL(2018) [[PDF]](https://www.aclweb.org/anthology/P18-1061) 
## 摘要生成
-  **Global Encoding**: "Global Encoding for Abstractive Summarization". ACL(2018)[[PDF]](https://aclanthology.org/P18-2027.pdf) [[code]](https://github.com/lancopku/Global-Encoding)
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
- **PubMed**: "Claim Extraction in Biomedical Publications using Deep Discourse Model and Transfer Learning". CoRR(2019) [[PDF]](http://arxiv.org/abs/1907.00962)[[code]](https://github.com/titipata/detecting-scientific-claim)
- **HoVer**: "HOVER: A Dataset for Many-Hop Fact Extraction And Claim Verification". EMNLP(2020)[[code]](https://doi.org/10.18653/v1/2020.findings-emnlp.309)
- **Fake-news-reasoning**: "Automatic Fake News Detection: Are Models Learning to Reason?". ACL/IJCNLP(2021) [[PDF]](https://doi.org/10.18653/v1/2021.acl-short.12) [[code]](https://github.com/casperhansen/fake-news-reasoning
)
- **ClaHi-GAT**: "Rumor Detection on Twitter with Claim-Guided Hierarchical Graph Attention Networks". EMNLP(2021) [[PDF]](https://doi.org/10.18653/v1/2021.emnlp-main.786)[[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/R9.Rumor%20Detection%20on%20Twitter%20with%20Claim-Guided%20Hierarchical%20Graph.pdf)
- **Group Learning**: "Students Who Study Together Learn Better: On the Importance of Collective Knowledge Distillation for Domain Transfer in Fact Verification". EMNLP(2021) [[PDF]](https://doi.org/10.18653/v1/2021.emnlp-main.558) [[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/R11.Students%20Who%20Study%20Together%20Learn%20Better%20On%20the%20Importance%20of.pdf)
- **Meet The Truth**: "Meet The Truth: Leverage Objective Facts and Subjective Views for Interpretable Rumor Detection". ACL/IJCNLP(Findings2021) [[PDF]](https://doi.org/10.18653/v1/2021.findings-acl.63)


## 情感分析
- **SC-CMC-KS**: "Sentiment classification model for Chinese micro-blog comments based on key sentences extraction". Soft Comput(2021) [[PDF]](https://link.springer.com/content/pdf/10.1007/s00500-020-05160-8.pdf)
- **nlp.stanford**: "Twitter Sentiment Analysis".Entropy(2009)  [[PDF]](https://www-nlp.stanford.edu/courses/cs224n/2009/fp/3.pdf)
- **3-Way**: "Twitter Sentiment Analysis, 3-Way Classification: Positive, Negative or Neutral?". IEEE BigData(2018) [[PDF]](https://doi.org/10.1109/BigData.2018.8621970) [[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/Twitter_sentiment_analysis_3_way_classification_positive_negative_or_neutral.pdf)
- **Twitter Data**: "Sentiment Analysis of Twitter Data". LSM(2011) [[PDF]](https://aclanthology.org/W11-0705.pdf) [[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/Sentiment%20analysis%20of%20twitter%20data.pdf)
- **STANKER**: "STANKER: Stacking Network based on Level-grained Attention-masked BERT for Rumor Detection on Social Media". EMNLP(2021) [[PDF]](https://doi.org/10.18653/v1/2021.emnlp-main.269) [[code]](https://github.com/fip-lab/STANKER) [[myNote]](https://github.com/lishuzhen97/Paper_reading/blob/main/Papers/R5.STANKER%20Stacking%20Network%20based%20on%20Level-grained%20Attention-masked.pdf)
- **SentiGAN**:"SentiGAN: Generating Sentimental Texts via Mixture Adversarial Networks". IJAC(2018) [[PDF]](https://doi.org/10.24963/ijcai.2018/618)
- **CNN**: "Convolutional Neural Networks for Sentence Classification". EMNLP() [[PDF]](https://doi.org/10.3115/v1/d14-1181)[[code]](https://code.google.com/p/word2vec/)
- **TBJE**: "A Transformer-based joint-encoding for Emotion Recognition and Sentiment Analysis". CoRR(2020) [[PDF]](https://arxiv.org/abs/2006.15955) [[code]](https://github.com/jbdel/MOSEI_UMONS)
-  ****

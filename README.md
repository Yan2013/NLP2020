# nlp_2020

## Assignment 2 NMT(neural machine translation)

因为pytorch有现成的tutorial, 故不设baseline.

- [TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [LANGUAGE TRANSLATION WITH TORCHTEXT](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html)

除tutorial给出的paper外, 推荐阅读:

- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)

## Dataset

数据集分为train/dev/test1/test2, 其中test2只有英文原文. 每一英文原文只有一句中文参考.  
link: https://www.kesci.com/home/dataset/5eb265a8366f4d002d761867

## Requirements

1. 本次NMT task为**英文翻译为中文**, 参考指标为BLEU(暂定).  
2. 最终请提交压缩包, 压缩包格式为`学号_姓名_NMT`, e.g. `SA19225001_张三_NMT`, 实验报告命名格式应该相同.
3. 压缩包的内容应该包括:
   - **实验报告**(pdf);
   - **源码**;
   - **test2数据集结果**(格式应和test1中文数据集相同). 
4. 实验报告至少应该包括:
   - 预处理过程, 模型结构, 超参数配置, 评估方法;
   - **test1**上的最终结果(非test2);
   - tensorboard 可视化训练结果;
   - 5个较低分数的翻译结果对比;
   - attention对比: soft, hard, global, local. 如果采用transformer架构, 请加入self-attention的对比

**ETA**  
请在5月25日晚22:00前发送压缩包至`zacb2018@mail.ustc.edu.cn`  

PS: 如有问题, 欢迎讨论.

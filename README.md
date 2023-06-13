# NLP_learning
一些传统NLP技术学习代码仓库

- 拼写纠错(斯坦福NLP公开课)
> 使用Markov Chain，结合贝叶斯概率，运用统计的方法实现英语单词的拼写纠错，该部分代码参考自github作者Rshcaroline的[Spell Correction项目](https://github.com/Rshcaroline/FDU-Natural-Language-Processing)，我在该代码上做了一些重构和修改。
- 基于机器学习的文本分类(Kaggle竞赛[Sentiment Analysis on Movie Reviews](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview))
> 使用N-Gram和Bag-of-words实现对自然语言文本的特征提取，使用numpy实现线性回归模型，并实验验证了使用shuffle、batch、mini-batch等权重更新方法的区别。
- 基于深度学习的文本分类(Kaggle竞赛[Sentiment Analysis on Movie Reviews](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview))
> 使用预训练的GloVe词向量提取文本特征，使用pytorch实现了TextCNN和TextRNN两种模型。TextCNN部分参考了论文[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)，TextRNN模型使用了RNN、LSTM、Bi-LSTM实现。

Loss函数新增OneVsAllLoss,用于多label分类，交叉熵。
Loss逻辑提取为独立的接口和类。

1.二进制文件格式貌似没有变化。

实现步骤：

1.读取c训练的分类模型，可以预测分类结果。
（多个模型和loss函数都需要测试）
2.读取c训练的压缩模型，可以预测分类。
3.java训练分类模型，可以获取分类结果，测试多种模型和损失函数。
4.转换c模型文本为java模型格式，测试结果。
5.java乘积量化模型压缩，测试分类结果。
6.测试词嵌入训练。
7.记录Fasttext源码角度的分析，词典、左右矩阵。


[新增OneVsAll损失函数] Introduction of the “OneVsAll” loss function for multi-label classification, which corresponds to the sum of binary cross-entropy computed independently for each label. This new loss can be used with the -loss ova or -loss one-vs-all command line option ( 8850c51 ).
【新增功能】Computation of the precision and recall metrics for each label ( be1e597 ).
Removed printing functions from FastText class ( 256032b ).
Better default for number of threads ( 501b9b1 ).【默认cpu数量应该是cpu核心数-1】


Bug fixes :
Normalize buffer vector in analogy queries.
Typo fixes and clarifications on website.
[完成]Fix: getSubwords for EOS.
[完成]Fix: ETA time.
Fix: division by 0 in word analogy evaluation. python的，无需修复。
Fix for the infinite loop on ARM cpu.


备注：
向量点积代表向量夹角。

input矩阵：
m x n :
m 代表词的数量
n 代表向量的维度
也就是m个词汇，每个词汇是n维度的向量

output矩阵
m 代表label的数量
n 代码向量的维度

hidden 向量
也就是把用户的一句话，查询input矩阵，对应里面的词向量，做加法，然后归一化。

分类预测：
简单来说：hidden 和 output求点积，计算出每个label的内积，然后
在softmax一下，最大的就是最大概率的label

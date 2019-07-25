# 注意力模型识别OCR

这个是基于[@thushv89](https://github.com/thushv89/attention_keras)的注意力模型的基础上，添加了OCR的功能，为了表示尊敬，直接fork了他。

但是，实际上我在他的基础上要添加OCR识别，参考的论文是
《Robust Scene Text Recognition with Automatic Rectification（RARE）》中的SRE网络。

### 跑一跑原作者的例子

为了加深attention的理解，可以跑原作者的Attention的例子：

`python -m examples.nmt_bidirectional.train`
要跑的话，需要先准备数据：
```
cd data
tar -xvf small_vocab_en.txt.gz
tar -xvf small_vocab_fr.txt.gz
```
---

关于K.rnn函数，读代码时候会遇到，参考[这个](https://kexue.fm/archives/5643/comment-page-1)
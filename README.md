# TR-experiments
这个repo已经有一万多行的Python代码了，每次做完实验，都没有做记录，昨天想了想，这样做完了实验什么都学不到，
之前忘记了这个readme的用途，今天开始记录所有的实验结果。留作以后分析
## 实验记录
### 2018年11月29日
* NGramSum：模型对当前单词的上下文单词进行求和，表示当前enrich过后的当前词，收到王少楠博士论文的启发，对上下文的词同样取一个重要性权重，具体方法：
设置一个位置embedding，当前词和位置embedding求出一个权重，然后乘以当前词作为表示。最后对上下文的词求和作为新的表示
这里上下文取了两种大小，一个是3，一个是5，都是以当前词为中心，然后两种上下文的词表示和当前词concat。
 ==============================***==============================
 CR : acc 0.7967, target 80.5000  
 MR : acc 0.7346, target 75.7000  
 TREC : acc 0.8712, target 89.0000  
 MPQA : acc 0.8290, target 83.7000  
 SST : acc 0.7973, target 81.4000  
 SUBJ : acc 0.9015, target 91.4000 
 
* 对于上下文窗口为5的，我们把上下文为3的上下文词(除去中心词剩下的词)去掉
 ==============================***==============================
 CR : acc 0.8063, epoch : 61, target 80.5000  
 MR : acc 0.7206, epoch : 187, target 75.7000  
 TREC : acc 0.8630, epoch : 101, target 89.0000  
 MPQA : acc 0.8439, epoch : 79, target 83.7000  
 SST : acc 0.7736, epoch : 162, target 81.4000  
 SUBJ : acc 0.8990, epoch : 158, target 91.4000  

* 之前对于拼接出上下文窗口的做法是：
    ```python
      x_embed = self.embedding(x)
      x_embed = self.dropout(x_embed)
      x_embed2 = self.embedding(torch.cat([torch.zeros(size = [x.shape[0], 1], dtype = torch.long).cuda(), x[:, 1:]], -1))
      x_embed3 = self.embedding(torch.cat([x[:, :-1], torch.zeros(size = [x.shape[0], 1], dtype = torch.long).cuda()], -1))
      x_embed4 = self.embedding(torch.cat([torch.zeros(size = [x.shape[0], 2], dtype = torch.long).cuda(), x[:, 2:]], -1))
      x_embed5 = self.embedding(torch.cat([x[:, :-2], torch.zeros(size = [x.shape[0], 2], dtype = torch.long).cuda()], -1))
    ```
  上述代码有问题：在前面补0应该截去原序列的后面，反之截去原序列的前面，上面正好相反。
    
**修改了问题**

* 收尾相接(不用0补充)，不挖空的结果：
 shouwei buwakong
==============================***==============================
 CR : acc 0.7912, epoch : 83, target 80.5000  	 
 MR : acc 0.7317, epoch : 91, target 75.7000  	 
 TREC : acc 0.8822, epoch : 187, target 89.0000  	 
 MPQA : acc 0.8141, epoch : 2, target 83.7000  	 
 SST : acc 0.7973, epoch : 177, target 81.4000  	 
 SUBJ : acc 0.9020, epoch : 75, target 91.4000  	
 
* 收尾相接，挖空：
 ==============================***==============================
 CR : acc 0.8036, epoch : 77, target 80.5000  	 
 MR : acc 0.7341, epoch : 110, target 75.7000  	 
 TREC : acc 0.8685, epoch : 170, target 89.0000  	 
 MPQA : acc 0.8290, epoch : 124, target 83.7000  	 
 SST : acc 0.7939, epoch : 42, target 81.4000  	 
 SUBJ : acc 0.9090, epoch : 64, target 91.4000

* 拼接0，挖空：
 ==============================***==============================
 CR : acc 0.7926, epoch : 97, target 80.5000  	 
 MR : acc 0.7346, epoch : 67, target 75.7000  	 
 TREC : acc 0.8795, epoch : 199, target 89.0000  	 
 MPQA : acc 0.8104, epoch : 2, target 83.7000  	 
 SST : acc 0.7979, epoch : 161, target 81.4000  	 
 SUBJ : acc 0.9025, epoch : 45, target 91.4000
 
 * 拼接0，不挖空：
 ==============================***==============================
 CR : acc 0.8036, epoch : 43, target 80.5000  	 
 MR : acc 0.7375, epoch : 71, target 75.7000  	 
 TREC : acc 0.8740, epoch : 85, target 89.0000  	 
 MPQA : acc 0.8141, epoch : 2, target 83.7000  	 
 SST : acc 0.8035, epoch : 197, target 81.4000  	 
 SUBJ : acc 0.9055, epoch : 25, target 91.4000  
 
 * 尝试使用王少楠文章中的方法，拼接上下文和当前单词的表示成为一个向量，使用一个中间矩阵，求出一个元素级别掩码门
 
 * 是否与词汇的频率有关系，扩展词典大小到20000，去掉对句子长度的限制(原先为6，现在为3)
 
 **GRU**
 ==============================***==============================
 
 |name|accuracy|epoch|target|
| :------| ------: | ------: |------: |
| CR |0.7793| 63 |80.5000  	 |
| MR |0.7529| 92 |75.7000  	 |
| TREC |0.8740| 107 |89.0000  	 |
| MPQA |0.8537| 103 |83.7000  	 |Y
| SST |0.8170| 192 |81.4000  	 |Y
| SUBJ |0.8955| 41 |91.4000|
 
 **NGramSumRNN**  	 
 sum one emb
==============================***==============================

|name|accuracy|epoch|target|
| :------| ------: | ------: |------: |
| CR |0.7779| 161 |80.5000  	 |
| MR |0.7605| 39 |75.7000  	 |Y
| TREC |0.8620| 99 |89.0000  	 |
| MPQA |0.8038| 34 |83.7000  	 |
| SST |0.8137| 145 |81.4000  	 |
| SUBJ |0.8945| 33 |91.4000 |
 
 two emb sum
==============================***==============================

|name|accuracy|epoch|target|
| :------| ------: | ------: |------: |
| CR |0.7846|93|80.5000  	 |
| MR |0.7572|32|75.7000  	 |Y
| TREC |0.8660|111|89.0000  	 |
| MPQA |0.7996|34|83.7000  	 |
| SST |0.7945|161|81.4000  	 |
| SUBJ |0.8910|58|91.4000  	 |

two emb max
==============================***==============================

|name|accuracy|epoch|target|
| :------| ------: | ------: |------: |
| CR |0.8019|62|80.5000  	 |
| MR |0.7647|158|75.7000  	 |Y
| TREC |0.8600|106|89.0000  	 |
| MPQA |0.8102|54|83.7000  	 |
| SST |0.8181|173|81.4000  	 |Y
| SUBJ |0.8985|71|91.4000  	|
 
 one emb max
==============================***==============================

|name|accuracy|epoch|target|
| :------| ------: | ------: |------: |
| CR| 0.7806| 167|80.5000  	 |
| MR| 0.7487| 191|75.7000  	 |
| TREC| 0.8660| 57|89.0000  	 |
| MPQA| 0.7953| 59|83.7000  	 |
| SST| 0.8192| 163|81.4000  	 |
| SUBJ| 0.8905| 17|91.4000  |

two pos emb shouwei
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7886| 69| 80.5000|
| MR | 0.7450| 120| 75.7000|
| TREC | 0.8720| 149| 89.0000|
| MPQA | 0.8123| 65| 83.7000|
| SST | 0.8214| 152| 81.4000| 
| SUBJ | 0.8940| 70| 91.4000|

## 2018年11月30日
* baseline lstm gather
baseline lstm gather
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7939| 41| 80.5000|  |
| MR | 0.7492| 96| 75.7000|  |
| TREC | 0.8780| 199| 89.0000|  |
| MPQA | 0.8621| 59| 83.7000| Y |
| SST | 0.8093| 166| 81.4000|  |
| SUBJ | 0.8960| 192| 91.4000|  |

baseline lstm maxpool
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7739| 61| 80.5000|  |
| MR | 0.7332| 175| 75.7000|  |
| TREC | 0.8480| 192| 89.0000|  |
| MPQA | 0.8770| 33| 83.7000| Y |
| SST | 0.7165| 4| 81.4000|  |
| SUBJ | 0.8730| 68| 91.4000|  |

baseline gru gather
==============================***==============================

|name   |accuracy       |epoch  |target |
| :------| ------: | ------: |------: |
| CR | 0.7832| 37| 80.5000|  |
| MR | 0.7562| 48| 75.7000|  |
| TREC | 0.8540| 185| 89.0000|  |
| MPQA | 0.8600| 65| 83.7000| Y |
| SST | 0.8110| 58| 81.4000|  |
| SUBJ | 0.8985| 196| 91.4000|  |


* 尝试把context的表示和当前词表示分开表示，在分类之前拼接

two gru
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7886| 90| 80.5000|  |
| MR | 0.7332| 39| 75.7000|  |
| TREC | 0.8480| 172| 89.0000|  |
| MPQA | 0.7826| 112| 83.7000|  |
| SST | 0.8132| 107| 81.4000|  |
| SUBJ | 0.8905| 29| 91.4000|  |

hiway 3 emb
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7899| 50| 80.5000|  |
| MR | 0.7421| 101| 75.7000|  |
| TREC | 0.8620| 104| 89.0000|  |
| MPQA | 0.7603| 137| 83.7000|  |
| SST | 0.7280| 39| 81.4000|  |
| SUBJ | 0.9010| 20| 91.4000|  |

* 代码去掉了kernel 和 stride判断，这一段是有问题的

modified lstm max
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7832| 74| 80.5000|  |
| MR | 0.7107| 38| 75.7000|  |
| TREC | 0.8460| 198| 89.0000|  |
| MPQA | 0.8738| 42| 83.7000| Y |
| SST | 0.8077| 183| 81.4000|  |
| SUBJ | 0.8840| 106| 91.4000|  |

lstm max gather
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7686| 76| 80.5000|  |
| MR | 0.7605| 96| 75.7000| Y |
| TREC | 0.8640| 155| 89.0000|  |
| MPQA | 0.8727| 63| 83.7000| Y |
| SST | 0.8214| 100| 81.4000| Y |
| SUBJ | 0.8980| 85| 91.4000|  |

lstm max gather
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7872| 96| 80.5000|  |
| MR | 0.6242| 3| 75.7000|  |
| TREC | 0.8740| 182| 89.0000|  |
| MPQA | 0.8643| 80| 83.7000| Y |
| SST | 0.8154| 198| 81.4000| Y |
| SUBJ | 0.8940| 106| 91.4000|  |

gru one emb gather
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7766| 45| 80.5000|  |
| MR | 0.7562| 60| 75.7000|  |
| TREC | 0.7160| 174| 89.0000|  |
| MPQA | 0.8590| 43| 83.7000| Y |
| SST | 0.8231| 180| 81.4000| Y |
| SUBJ | 0.8905| 81| 91.4000|  |

GRU max
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7699| 127| 80.5000|  |
| MR | 0.7144| 128| 75.7000|  |
| TREC | 0.8720| 151| 89.0000|  |
| MPQA | 0.8537| 88| 83.7000| Y |
| SST | 0.8137| 187| 81.4000|  |
| SUBJ | 0.8640| 32| 91.4000|  |

only use one context gru max
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7779| 73| 80.5000|  |
| MR | 0.6923| 82| 75.7000|  |
| TREC | 0.8580| 195| 89.0000|  |
| MPQA | 0.8685| 35| 83.7000| Y |
| SST | 0.7813| 131| 81.4000|  |
| SUBJ | 0.8620| 106| 91.4000|  |

使用了Ngram5作为context
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7872| 134| 80.5000|  |
| MR | 0.6275| 2| 75.7000|  |
| TREC | 0.8840| 197| 89.0000|  |
| MPQA | 0.8621| 162| 83.7000| Y |
| SST | 0.8269| 196| 81.4000| Y |
| SUBJ | 0.8995| 108| 91.4000|  |

使用了gram3作为context
==============================***==============================

|name	|accuracy	|epoch	|target	|
| :------| ------: | ------: |------: |
| CR | 0.7939| 59| 80.5000|  |
| MR | 0.6261| 5| 75.7000|  |
| TREC | 0.8680| 169| 89.0000|  |
| MPQA | 0.8568| 41| 83.7000| Y |
| SST | 0.8159| 188| 81.4000| Y |
| SUBJ | 0.8950| 44| 91.4000|  |

2018年12月01日

```python
    def forward(self, x, y):
        max_len = x.shape[1]
        x = LongTensor(x)
        mask = torch.where(x > 0, torch.ones_like(x, dtype = torch.float32), torch.zeros_like(x, dtype = torch.float32))
        x_embed = self.embedding(x)
        x_embed = self.dropout(x_embed)
        # x_embed2 = self.embedding(torch.cat([x[:, :1], x[:, :-1]], -1))
        # x_embed3 = self.embedding(torch.cat([x[:, 1:], x[:, :1]], -1))
        # x_embed4 = self.embedding(torch.cat([x[:, -2:], x[:, :-2]], -1))
        # x_embed5 = self.embedding(torch.cat([x[:, 2:], x[:, :2]], -1))

        x_embed2 = self.embedding(torch.cat([torch.zeros(size = [x.shape[0], 1], dtype = torch.long).cuda(), x[:, :-1]], -1))
        x_embed3 = self.embedding(torch.cat([x[:, 1:], torch.zeros(size = [x.shape[0], 1], dtype = torch.long).cuda()], -1))
        x_embed4 = self.embedding(torch.cat([torch.zeros(size = [x.shape[0], 2], dtype = torch.long).cuda(), x[:, :-2]], -1))
        x_embed5 = self.embedding(torch.cat([x[:, 2:], torch.zeros(size = [x.shape[0], 2], dtype = torch.long).cuda()], -1))
        #
        pos1 = self.pos_embedding(LongTensor([0, 1, 2]))
        pos2 = self.pos_embedding2(LongTensor([3, 0, 1, 2, 4]))

        ngram3 = torch.stack([x_embed2, x_embed, x_embed3], -2).sum(2).squeeze(2)
        ngram5 = torch.stack([x_embed4, x_embed2, x_embed, x_embed3, x_embed5], -2).sum(2).squeeze(2)
        # ngram3 = F.softmax(torch.sum(ngram3 * pos1, -1), -2).unsqueeze(-1) * ngram3
        # ngram5 = F.softmax(torch.sum(ngram5 * pos2, -1), -2).unsqueeze(-1) * ngram5

        # x_embed = torch.cat([ngram3.max(2)[0], ngram5.max(2)[0], x_embed], -1)
        # x_embed = torch.cat([x_embed, ngram3.sum(2).squeeze(2), ngram5.sum(2).squeeze(2)], -1)
        ngram3_s = F.sigmoid(torch.cat([ngram3, ngram5, x_embed], -1) @ self.param) * ngram3
        x_embed = x_embed + ngram3_s * F.tanh(ngram3) + (1-ngram3_s) * F.tanh(ngram5)
        outputs, (h, c) = self.rnn1(x_embed)
        # output_maxpooled, _ = torch.max(outputs, 1)
        output_maxpooled = gather_rnnstate(outputs, mask).squeeze(1)
        class_prob = self.linear(output_maxpooled)
        return class_prob, F.dropout(output_maxpooled)

```
==============================***==============================

|name   |accuracy       |epoch  |target |
| :------| ------: | ------: |------: |
| CR | 0.7766| 73| 80.5000|  |
| MR | 0.7628| 68| 75.7000| Y |
| TREC | 0.8680| 194| 89.0000|  |
| MPQA | 0.8600| 24| 83.7000| Y |
| SST | 0.8110| 196| 81.4000|  |
| SUBJ | 0.9025| 59| 91.4000|  |

比baseline gru 在 MR TREC MPQA 强 SST相等， 在CR弱
比baseline lstm 在 MR SST TREC 强，其他都弱

|name   |accuracy       |epoch  |target |
| :------| ------: | ------: |------: |
| CR | 0.7859| 85| 80.5000|  |
| MR | 0.7581| 108| 75.7000| Y |
| TREC | 0.8760| 126| 89.0000|  |
| MPQA | 0.8484| 181| 83.7000| Y |
| SST | 0.8165| 91| 81.4000| Y |
| SUBJ | 0.9015| 188| 91.4000|  |
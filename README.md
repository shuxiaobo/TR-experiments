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
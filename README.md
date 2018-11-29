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
 
   
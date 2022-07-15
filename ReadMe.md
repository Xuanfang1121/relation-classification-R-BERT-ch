### R-Bert chinese
论文为Enriching Pre-trained Language Model with Entity Information for Relation Classification
R-BERT是关系分类，本代码是在中文数据集上的应用。

#### 依赖
```
transformers==4.10.1
torch==1.8.1
numpy==1.19.2
scikit_learn==1.1.1
```

#### 代码结构
```
  src
   |__common
   |__config
   |__models
   |__utils
   |__train.py
   |__predict.py

```

#### 运行
1. 修改config/config.ini下的相关参数 <br>
2. 模型训练 python train.py <br>
3. 模型预测 python predict.py 


#### 实验结果
```
1. 在一个关系分类数据集上结果如下， 关系数量为12个
                 precision    recall  f1-score   support

        unknown     0.99      0.99      0.99     19630
          父母       0.98      0.99      0.99      7220
          夫妻       0.99      0.99      0.99      9104
          师生       0.97      0.96      0.97      2100
        兄弟姐妹       0.99      0.99      0.99      3007
          合作       0.98      0.99      0.98      3664
          情侣       1.00      0.99      0.99      2120
          祖孙       0.99      0.96      0.98       498
          好友       0.99      0.97      0.98       770
          亲戚       0.98      0.96      0.97       510
          同门       0.98      0.98      0.98       709
         上下级       0.99      0.99      0.99       668

    accuracy                           0.99     50000
   macro avg       0.99      0.98      0.98     50000
weighted avg       0.99      0.99      0.99     50000

acc:0.98826, recall:0.9800481728287208, precision:0.9865507605007057, f1:0.9832653815196769
```

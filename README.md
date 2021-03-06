# lowsale_dnn_predict
长尾商品销量DNN预测

长尾商品的预测是一个难题，目前尝试了很多方法。
这里只考虑7天、30天的预测和60天的预测，来服务于供应链系统的备货。

## 1. 考虑单输出，如7天预测一个模型，30天预测一个模型
- 基于单纯的销量，possion分布在7天预测上的表现还可以，不过有相应的前提条件。     
1） 销量数据不能超过一定阀值，否则possion分布预测失效   
2） 需要check下哪段范围内的销量均值作为lambda的取值，否则预测出来的值就不准确
3） 另外还需要check下概率p值，多大概率下销量的预测值较为合适     
这里没有考虑lstm等dnn模型，主要是稀疏数据上无法学习相关隐含关系。另外对于销量超过一定阀值的数据，需要借助销量指数衰减加权来补充预测
- 摆脱单纯的销量，考虑除销量外的其他特征，使用DNN和lgb进行预测              
1）在有特征情况下，lgb的使用是非常自然的，代码过程非常简单易懂，唯一的难点在于调参和特征工程，特征工程对于使用DNN和lgb都是前置基础，都需要做足够多
的基础特征，唯一的差异在于DNN可以帮助做很多特征组合方面的工作，lgb的调参，借助optuna完成自动调参，自动调参的时间比较长，不过减少人为参与。
另外用训练好的lgb模型检测其他时间的预测稍准，lgb模型的稳定性一定程度上还行，适合短期预测，7天预测还可以，30天预测相比dnn还是差很多。              
2）基于dnn的预测，目前在30天的预测上表现比较好；由于数据是按天整理的销量数据和相应的特征，使用一天或者少数几天的数据进行训练，泛化能力比较欠缺，
分析原因，在于样本分布的差异，目前使用更多天数的数据进行训练，预测效果有了显著提升。

目前提供低阶的tensrflow api来进行长尾数据的训练和预测，tensorflow==1.13， 包含如下：
1）由于要使用tensorboard， 因此tf.summary.FileWriter的使用方法， 启动方式                   
`tensorboard --logdir=./tmp/`                      
2）模型的保存，使用tf.train.Saver            
3）训练、测试和验证集的划分。       
训练集：用来训练模型                     
验证集：用来选择超参数                          
测试集：评估模型的泛化能力                        
4）关于earlystop的自定义实现，需要注意earlystop的本质： 记录到目前为止最好的验证集精度，当连续10次Epoch（或者更多次）没达到最佳精度时，
则可以认为精度不再提高了。               
5）关于离线模型的加载和预测，使用tf.train.import_meta_graph和graph.get_operation_by_name来进行预测

## 2. 考虑多输出，一个模型同时优化7天预测、30天预测和60天预测：        
核心代码如下：  
```
    _ = keras.layers.Dense(num_units[2], activation='relu')(drop3)
    week_predict = keras.layers.Dense(output_size, name="week_predict")(_)
    _ = keras.layers.Dense(num_units[2], activation='relu')(drop3)
    month_predict = keras.layers.Dense(output_size, name="month_predict")(_)
    _ = keras.layers.Dense(num_units[2], activation='relu')(drop3)
    two_month_predict = keras.layers.Dense(output_size, name="two_month_predict")(_)

    model = keras.models.Model(inputs=[input_x], outputs=[week_predict, month_predict, two_month_predict])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
        loss={'week_predict': 'mae', 'month_predict': 'mae', 'two_month_predict': 'mae'},
        loss_weights={'week_predict': 4., 'month_predict': 2, 'two_month_predict': 1.},
        metrics={'week_predict': root_mean_squared_error,
                 'month_predict': root_mean_squared_error,
                 'two_month_predict': root_mean_squared_error}
        )
```

目前线上使用的是多输出的预测，效果显著！

# 仅用于学习日记，不可用于任何商业用途
#利用GBDT+LR进行预测 facebook提出
#1、整理数据，将文字用数字表示
#2、分割数据 sklearn.model_selection.train_test_split
#3、建立模型和训练
#先建立gbdt模型和训练数据  sklearn.ensemble.GradientBoostingClassifier   gbdt.fit(x_train,y_train)
#将gbdt训练好的数据onehot化  OneHotEncoder.fit(gbdt.apply(x_train).reshape(-1,100)) 一定要reshape为2维
#将onehot化的数据传给LR训练  lr.fit(OneHotEncoder.transform(gbdt.apply(x_train).reshape(-1,100),y_train)
#4、预测
#可以用lr.predict 自带的sigmoid 生成0/1分布
#也可以用lr.predict_prob 生成概率在转为0/1
#sklearn.metrics.mean_squared_error（y_predict,y_test) 算出mse
#sklearn.metrics.accuracy_score(y_predict,y_test) 算出正确率


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:49:50 2019

@author: liujun
"""

import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier #支持多分类
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder



class basedGBDT_LR: 
    def __init__(self):
        self.filepath='telecom-churn-prediction-data.csv'
        self.data=self.feature_transform()
        self.train,self.test=self.split_data()
        self.lr,self.gbdt,self.enc=self.model()
        
    def isnone(self,value):
        if value==" " or value is None:
            return '0'
        else:
            return value
        
    def feature_transform(self):
        if os.path.exists('new_data.csv'):
            return pd.read_csv('new_data.csv')
        else:
            print('转换特征值')
            feature_dict={
                    'gender':{'Male':'1','Female':'0'},
                    'Partner':{'Yes':'1','No':'0'},
                    'Dependents':{'Yes':'1','No':'0'},
                    'PhoneService':{'Yes':'1','No':'0'},
                    'MultipleLines':{'Yes':'1','No':'0','No phone service':'2'},
                    'InternetService':{'DSL':'1','Fiber optic':'2','No':'0'},
                    'OnlineSecurity':{'Yes':'1','No':'0','No internet service':'2'},
                    'OnlineBackup':{'Yes':'1','No':'0','No internet service':'2'},
                    'DeviceProtection':{'Yes':'1','No':'0','No internet service':'2'},
                    'TechSupport':{'Yes':'1','No':'0','No internet service':'2'},
                    'StreamingTV':{'Yes':'1','No':'0','No internet service':'2'},
                    'StreamingMovies':{'Yes':'1','No':'0','No internet service':'2'},
                    'Contract':{'Month-to-month':'0','One year':'1','Two year':'2'},
                    'PaperlessBilling':{'Yes':'1','No':'0'},
                    'PaymentMethod':{'Electronic check':'0','Mailed check':'1','Bank transfer (automatic)':'2','Credit card (automatic)':'3'},
                    'Churn':{'Yes':'1','No':'0'},}
            fw=open('new_data.csv','w')
            fw.write("customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\n")
            for line in open(self.filepath,'r').readlines():
                if not line.startswith('customerID'):
                    customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn=line.strip().split(',')
                    data=[]
                    data.append(customerID) #如果ID空缺  则不要本条数据
                    data.append(self.isnone(feature_dict['gender'][gender]))
                    data.append(self.isnone(SeniorCitizen))
                    data.append(self.isnone(feature_dict['Partner'][Partner]))
                    data.append(self.isnone(feature_dict['Dependents'][Dependents]))
                    data.append(self.isnone(tenure))
                    data.append(self.isnone(feature_dict['PhoneService'][PhoneService]))
                    data.append(self.isnone(feature_dict['MultipleLines'][MultipleLines]))
                    data.append(self.isnone(feature_dict['InternetService'][InternetService]))
                    data.append(self.isnone(feature_dict['OnlineSecurity'][OnlineSecurity]))
                    data.append(self.isnone(feature_dict["OnlineBackup"][OnlineBackup]))
                    data.append(self.isnone(feature_dict["DeviceProtection"][DeviceProtection]))
                    data.append(self.isnone(feature_dict["TechSupport"][TechSupport]))
                    data.append(self.isnone(feature_dict["StreamingTV"][StreamingTV]))
                    data.append(self.isnone(feature_dict["StreamingMovies"][StreamingMovies]))
                    data.append(self.isnone(feature_dict["Contract"][Contract]))
                    data.append(self.isnone(feature_dict["PaperlessBilling"][PaperlessBilling]))
                    data.append(self.isnone(feature_dict["PaymentMethod"][PaymentMethod]))
                    data.append(self.isnone(MonthlyCharges))
                    data.append(self.isnone(TotalCharges))
                    data.append(self.isnone(feature_dict["Churn"][Churn]))
                    fw.write(','.join(data))
                    fw.write('\n')
            return pd.read_csv('new_churn.csv')
    
    def split_data(self): #分割数据
        train,test=train_test_split(self.data,test_size=0.2,random_state=4)
        return train,test
    
    def model(self): #x为除去customerid和churn的值  有y为churn
        x_train=self.train[[x for x in self.train.columns if x not in ['customerID','Churn']]]
        y_train=self.train['Churn']
        lr=LogisticRegression(penalty='l2',tol=0.0001,fit_intercept=True,max_iter=20)
        gbdt=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=7) #学习率为0.1 最大迭代次数为100，树深为7
        gbdt.fit(x_train,y_train)
        #gbdt产生的是3维数据
        enc=OneHotEncoder()
        enc.fit(gbdt.apply(x_train).reshape(-1,100))
        lr.fit(enc.transform(gbdt.apply(x_train).reshape(-1,100)),y_train)
        return lr,gbdt,enc
     
    def evaluate(self):
        x_test=self.test[[x for x in self.test.columns if x not in ['customerID','Churn']]]
        y_test=self.test['Churn']
        y_predict=self.lr.predict_proba(self.enc.transform(self.gbdt.apply(x_test).reshape(-1,100)))
        y_predict_list=[]
        for y in y_predict:
            y_predict_list.append(1 if y[1]>0.5 else 0) #回归概率大于0.5为流失 反之为不流失
        mse=mean_squared_error(y_test,y_predict_list) #平均方差
        print('mse:{}'.format(mse))
        accuracy=metrics.accuracy_score(y_test.values,y_predict_list) #正确率
        print('accuracy:{}'.format(accuracy))
        auc=metrics.roc_auc_score(y_test.values,y_predict_list)
        print('auc:{}'.format(auc))
        
if __name__ == "__main__":
    pred = basedGBDT_LR()
    pred.evaluate()
        
        
                   
                    
                    
        
        
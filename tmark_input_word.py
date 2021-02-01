# get_ipython().system('pip install animation')
# get_ipython().system('pip install sklearn')
# get_ipython().system('pip install numpy')
# get_ipython().system('pip install matplotlib')
# get_ipython().system('pip install flask')
# get_ipython().system('pip install linebot-bot-sdk')
# get_ipython().system('pip install flask_ngrok')

#安裝line bot flask相關套件
import json , os , shutil , base64 , requests , logging , re
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn import decomposition
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import jieba
from sklearn import cluster, datasets, metrics
import animation

input_time = time.time()

def find_same_word(input_s, g_codes, db_path):
    df      = pd.read_csv(db_path,sep=",",header=None)
    df_gcode  = df.iloc[:,2]
    df_name   = df.iloc[:,1]
#     print(df_gcode)
    
#先找出同類別的商標們
    g_list = []
    no1_list = []
    n = 0
    a = len(df_gcode)
    for i in range(a):
        n += 1
        try:
            if str(df_gcode[i]) == str(g_codes):
                if df_name[i] not in g_list:
                    g_list.append(str(df_name[i]))                    
        except:
            pass
# #有一樣字的就選出來 
# #把問題也加入 list 
    output_list = [input_s]
    
    for j in str(input_s):
        b = len(g_list)
        for i in range(b): 
            c = len(str(g_list[i]))
            for k in range(c):
                if str(j) == str(g_list[i])[k]:
                    if str(g_list[i]) not in output_list:
                        output_list.append(str(g_list[i]))    
    new_df = pd.DataFrame(output_list,columns=["output"])   
#     new_df.to_csv("./data/input_temp.csv",index=False)

#2.one hot encoding
#     df_input = pd.read_csv("./data/input_temp.csv")
#     df = df_input['output']
    df = new_df['output']
    out_put_list = []
    for i in range(len(df)):
        out_put_one_list = []
        for j in df[i]:
            out_put_one_list.append(j)
        out_put_list.append(out_put_one_list)   
    out_put_df = pd.DataFrame(out_put_list) 
    dummy_df = pd.get_dummies(out_put_df)
    df_train_one_sk = pd.DataFrame(OneHotEncoder().fit_transform(dummy_df).toarray())
    array_train_one_sk = OneHotEncoder().fit_transform(dummy_df).toarray()
    
#3.PCA降維
    X_pca = array_train_one_sk
    pca = decomposition.PCA(n_components=5)
    X_pca_done = pca.fit_transform(X_pca)
    X_pca_df = pd.DataFrame(X_pca_done)
    
#4. 開始各種分群
    model_a1 = AgglomerativeClustering(n_clusters=15, linkage='average')
    c_a1 = model_a1.fit_predict(X_pca_done)
    label_a1 = pd.Series(model_a1.labels_)

    model_a2 = AgglomerativeClustering(n_clusters=15, linkage='complete')
    c_a2 = model_a2.fit_predict(X_pca_done)
    label_a2 = pd.Series(model_a2.labels_)

    model_a3 = AgglomerativeClustering(n_clusters=15, linkage='ward')
    c_a3 = model_a3.fit_predict(X_pca_done)
    label_a3 = pd.Series(model_a3.labels_)

    model_a4 = AgglomerativeClustering(n_clusters=15, linkage='single')
    c_a4 = model_a4.fit_predict(X_pca_done)
    label_a4 = pd.Series(model_a4.labels_)

    model_k1 = KMeans(n_clusters=15, init="random")
    c_k1 = model_k1.fit_predict(X_pca_done)
    label_k1 = pd.Series(model_k1.labels_)

    model_k2 = KMeans(n_clusters=15, init="k-means++")
    c_k2 = model_k2.fit_predict(X_pca_done)
    label_k2 = pd.Series(model_k2.labels_) 
    
    clus_df = pd.DataFrame()
    clus_df["ag_average"] = label_a1
    clus_df["ag_complete"] = label_a2
    clus_df["ag_ward"]   = label_a3
    clus_df["ag_single"]  = label_a4
    clus_df["kmeans"]    = c_k1
    clus_df["kmeans_plus"] = c_k2
    
    out_put_name_list = []
    for i in out_put_list:
        name = str(i).strip("[]").replace("'","").replace(", ","")
        out_put_name_list.append(name)
    df_n = pd.DataFrame(out_put_name_list)
    df_new = pd.concat([df_n,clus_df],axis=1)
#     df_new.to_csv("./data/trained.csv",encoding="utf-8-sig")
    
    
    #5. 和問題同群的有
    q_a_list = []
    for i in df_new.columns:
        q_a_list.append(df_new[i][0])
    all_list = []

    for j in range(1,len(q_a_list)):
        one_list = []
        one_no_list = []
        for k in range(1,len(df_new)):
            if str(df_new.iloc[:,j][k]) == str(q_a_list[j]):
                one_list.append(df_new.iloc[:,0][k])
                one_no_list.append(df_new.iloc[:,-1][k])
        all_list.append(one_list) 

    all_df = pd.DataFrame(all_list)
    
#6. 分群結果計算分數
    silhouette_a1 = metrics.silhouette_score(X_pca_done, label_a1)
    silhouette_a2 = metrics.silhouette_score(X_pca_done, label_a2)
    silhouette_a3 = metrics.silhouette_score(X_pca_done, label_a3)
    silhouette_a4 = metrics.silhouette_score(X_pca_done, label_a4)
    silhouette_k1 = metrics.silhouette_score(X_pca_done, c_k1)
    silhouette_k2 = metrics.silhouette_score(X_pca_done, c_k2)
    silhouette_score_list = [silhouette_a1,silhouette_a2,silhouette_a3,silhouette_a4,silhouette_k1,silhouette_k2]
#     print(silhouette_score_list)
    n = len(silhouette_score_list)
    silhouette_percentage = []

    for i in silhouette_score_list:
        silhouette_percentage.append(float("{:.2f}".format((i/n)*100)))
#     print(silhouette_percentage )


    q_a_list = []
    for i in df_new.columns:
        q_a_list.append(df_new[i][0])
    unique_list = []    
    for j in range(1,len(q_a_list)):
    #     one_list = []
        for k in range(1,len(df_new)):
            if str(df_new.iloc[:,j][k]) == str(q_a_list[j]):
                if df_new.iloc[:,0][k] not in unique_list:
                    if df_new.iloc[:,0][k] != None:
                        unique_list.append(df_new.iloc[:,0][k])

    score_list = [0]*len(unique_list)
    for i in range(len(unique_list)):
        if unique_list[i] in all_list[0]:
            score_list[i] += silhouette_percentage[0]            
        if unique_list[i] in all_list[1]:
            score_list[i] += silhouette_percentage[1]
        if unique_list[i] in all_list[2]:
            score_list[i] += silhouette_percentage[2]
        if unique_list[i] in all_list[3]:
            score_list[i] += silhouette_percentage[3]
        if unique_list[i] in all_list[4]:
            score_list[i] += silhouette_percentage[4]
        if unique_list[i] in all_list[5]:
            score_list[i] += silhouette_percentage[5]
            
    df_score = pd.DataFrame()

    df_score["name"] = unique_list
    df_score["score"] = score_list

    ans = df_score.sort_values(by="score",ascending=False).head(5)
    return ans, df_train_one_sk, df_new 


# 原本            
# if __name__ == "__main__":
#     A = r"春天"
#     B = r"032"
#     C = r"./drive/MyDrive/online/g_code_tname_clean.csv"   #跟改資料路徑
#     ans,x,y_df = find_same_word(A,B,C)
#     print(ans)
    



# # 測試
# if __name__ == "__main__":
#     A = input('輸入來源 : ' + "")
#     B = input('輸入類別 : ' + "")
#     C = r"./g_code_tname_clean.csv"   #跟改資料路徑
#     ans,x,y_df = find_same_word(A,B,C)

#     print(ans)



#151s


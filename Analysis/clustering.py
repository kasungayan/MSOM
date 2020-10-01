import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.cluster import KMeans

df_all = pd.read_csv("Preprocessed_data/UON_reorganised_data-V9.csv")

click_data_withsku = ['clicktable-sku_ID','clicktable-request_time', 'clicktable-channel', 'skutable-type','skutable-attribute1', 'skutable-attribute2', 'usertable-user_level',
                     'usertable-plus', 'usertable-gender', 'usertable-age', 'usertable-marital_status', 'usertable-education', 'usertable-city_level',
                      'usertable-purchase_power', 
                      'ordertable-quantity', 
                      'ordertable-type', 
                      'ordertable-promise', 
                      'ordertable-number_of_gifts',
                      'ordertable-original_unit_price',
                      'ordertable-direct_discount_per_unit',
                      'ordertable-quantity_discount_per_unit',
                      'ordertable-bundle_discount_per_unit',
                      'ordertable-coupon_discount_per_unit', 
                      'newtable-lastorderquantity']
                      
click_data_df_version1 = df_all[click_data_withsku]
click_data_df_version2_withoutna = click_data_df_version1.dropna()
click_data_df_version2_withoutna_withoutdash = click_data_df_version2_withoutna.replace({'-': None})
 
click_data_df_version2_withoutna_withoutdash = click_data_df_version2_withoutna_withoutdash.fillna(value=np.nan)
 
click_data_df_version2_withoutna_withoutdash = click_data_df_version2_withoutna_withoutdash.dropna()
 
click_data_df_version2_withoutna_withoutdash['ordertable-quantity'] = click_data_df_version2_withoutna_withoutdash['ordertable-quantity'].replace({'nilorder': 0})
 
click_data_df_version2_withoutna_withoutdash.rename(columns = {'ordertable-quantity':'quantity'},inplace = True)
 
df_all = click_data_df_version2_withoutna_withoutdash.copy()
 
df_all['quantity'] = pd.to_numeric(df_all['quantity'], errors = 'coerce') 
 
df_all_filter = df_all.dropna()
 
df_final = df_all_filter.loc[df_all_filter['quantity'] > 0,]
 
df_final.reset_index(drop=True, inplace=True)
 
df_products = pd.read_csv("JD_sku_data.csv")
 
df_products_feature = df_products[['sku_ID','attribute1', 'attribute2']]
df_products_feature_vector = df_products_feature.replace({'-': None})
df_products_feature = df_products_feature_vector.fillna(value=np.nan)
df_products_feature_final = df_products_feature.dropna()
 
sku_id = df_products_feature_final['sku_ID']
df_cluster_feature_vector = df_products_feature_final[['attribute1', 'attribute2']]
 
scaler = MinMaxScaler()
df_cluster_feature_vector[['attribute1', 'attribute2']] = scaler.fit_transform(df_cluster_feature_vector[['attribute1', 'attribute2']])
 
distortions = []
K = range(1,10)

for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_cluster_feature_vector)
    distortions.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

kmeanModel = KMeans(n_clusters=4)
kmeanModel.fit(df_cluster_feature_vector)

df_cluster_feature_vector['class']=kmeanModel.predict(df_cluster_feature_vector)

y_kmeans = df_cluster_feature_vector['class']

y_kmeans = np.array(y_kmeans)

X = df_cluster_feature_vector

#6 Visualising the clusters
plt.scatter(X.loc[y_kmeans==0, 'attribute1'], X.loc[y_kmeans==0, 'attribute2'], s=100, c='red', label ='Cluster 1')
plt.scatter(X.loc[y_kmeans==1, 'attribute1'], X.loc[y_kmeans==1, 'attribute2'], s=100, c='blue', label ='Cluster 2')
plt.scatter(X.loc[y_kmeans==2, 'attribute1'], X.loc[y_kmeans==2, 'attribute2'], s=100, c='green', label ='Cluster 3')
plt.scatter(X.loc[y_kmeans==3, 'attribute1'], X.loc[y_kmeans==3, 'attribute2'], s=100, c='cyan', label ='Cluster 4')
#plt.scatter(X.loc[y_kmeans==4, 'attribute1'], X.loc[y_kmeans==4, 'attribute2'], s=100, c='cyan', label ='Cluster 5')
#Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
plt.scatter(kmeanModel.cluster_centers_[:, 0], kmeanModel.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.title('Clusters of Products')
plt.xlabel('Atrribute 1')
plt.ylabel('Atrribute 2')
plt.show()


final_df = pd.concat([sku_id, df_cluster_feature_vector], axis = 1)

df_final.rename(columns = {'clicktable-sku_ID':'sku_ID'}, inplace = True)

df_final['sku_ID'] = df_final['sku_ID'].astype(str)
final_df['sku_ID'] = final_df['sku_ID'].astype(str)

df_final_joined = df_final.merge(final_df, on = "sku_ID", how = "inner")

#df_final_joined['class'].value_counts()

df_final_joined = df_final.copy()
click_data_df_v2 = df_final_joined.copy()

click_data_df_v2['clicktable-request_time'] = pd.to_datetime(click_data_df_v2['clicktable-request_time'])
click_data_df_v2.rename(columns = {'clicktable-request_time':'Time'},inplace = True) #

click_data_df_v2['day_of_week']= click_data_df_v2.Time.dt.weekday
click_data_df_v2['hour_of_day'] = click_data_df_v2.Time.dt.hour

click_data_df_v2 = click_data_df_v2.drop(columns = ['Time', 'ordertable-type','newtable-lastorderquantity'])

click_data_df_v2['ordertable-original_unit_price'] = pd.to_numeric(click_data_df_v2['ordertable-original_unit_price'], errors = 'coerce')
click_data_df_v2['ordertable-direct_discount_per_unit'] = pd.to_numeric(click_data_df_v2['ordertable-direct_discount_per_unit'], errors = 'coerce') 
click_data_df_v2['ordertable-quantity_discount_per_unit'] = pd.to_numeric(click_data_df_v2['ordertable-quantity_discount_per_unit'], errors = 'coerce') 
click_data_df_v2['ordertable-bundle_discount_per_unit'] = pd.to_numeric(click_data_df_v2['ordertable-bundle_discount_per_unit'], errors = 'coerce') 
click_data_df_v2['ordertable-coupon_discount_per_unit'] = pd.to_numeric(click_data_df_v2['ordertable-coupon_discount_per_unit'], errors = 'coerce') 

shap_v1 = click_data_df_v2.copy()

shap_v1['percentage_direct_discount_per_unit'] = shap_v1['ordertable-direct_discount_per_unit']/shap_v1['ordertable-original_unit_price']
shap_v1['percentage_direct_discount_per_unit'] = shap_v1['ordertable-direct_discount_per_unit']/shap_v1['ordertable-original_unit_price']
shap_v1['percentage_quantity_discount_per_unit'] = shap_v1['ordertable-quantity_discount_per_unit']/shap_v1['ordertable-original_unit_price']
shap_v1['percentage_bundle_discount_per_unit'] = shap_v1['ordertable-bundle_discount_per_unit']/shap_v1['ordertable-original_unit_price']
shap_v1['percentage_coupon_discount_per_unit'] = shap_v1['ordertable-coupon_discount_per_unit']/shap_v1['ordertable-original_unit_price']

shap_v1 = shap_v1.drop(columns = ['ordertable-direct_discount_per_unit', 'ordertable-quantity_discount_per_unit', 'ordertable-bundle_discount_per_unit','ordertable-coupon_discount_per_unit'])

shap_v1.rename(columns = {'usertable-age':'age', 
                          'clicktable-sku_ID': 'skuID',
                         'ordertable-promise':'promise',
                          'usertable-user_level':'user_level',
                          'usertable-education':'education',
                          'usertable-city_level':'city_level',
                          'usertable-purchase_power':'purchase_power',
                          'usertable-gender':'gender',
                          'clicktable-channel':'channel',
                          'usertable-marital_status':'marital_status'
                         }, inplace = True)
                         
shap_v1.rename(columns = {'skutable-type':'product_type',
                          'usertable-plus':'user_plus',
                          'ordertable-number_of_gifts':'number_of_gifts'
                         }, inplace = True)
                         
                         
                         
shap_v1.reset_index(drop=True, inplace=True)
#shap_v1.shape

shap_v1.to_feather('click_data_classification_correct_approach1.ft')





 
 
 
 
 
 
 


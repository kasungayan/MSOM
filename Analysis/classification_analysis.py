import pandas as pd
import numpy as np
import plotly
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

import pickle

np.random.seed(123)

df_final_joined = pd.read_feather('Preprocessed_data/click_data_classification_correct_approach1.ft')

shap_v1_filter = df_final_joined.copy()
shap_v1_filter['class'] = shap_v1_filter['class'].astype('int8')

shap_v1_filter = shap_v1_filter.drop(columns = ['sku_ID','skutable-attribute1', 'skutable-attribute2','quantity', 'attribute1',
                                               'attribute2'])

shap_v1 = shap_v1_filter.copy()

age_categories = ['U','<=15', '16-25', '26-35', '36-45', '46-55', '>=56']
#ordertablepromise_categories = ['nilorder','1', '2', '3', '4', '5', '6', '7', '8']
#ordertable_gift_item_categories = ['nilorder','0', '1']
#newtable_hasdiscount_categories = ['nilnewtable','yes', 'no']
userlevel_categories = [-1, 0,1, 2, 3, 4, 10]
education_categories = [-1, 1, 2, 3, 4]
citylevel_categories = [-1, 1, 2, 3, 4, 5]
purchastingpower_categories = [-1, 1, 2, 3, 4, 5]

shap_v1["age"] = shap_v1["age"].astype("category", ordered=True,categories=age_categories).cat.codes
#shap_v1["promise"] = shap_v1["promise"].astype("category", ordered=True,categories=ordertablepromise_categories).cat.codes
#shap_v1["ordertable-gift_item"] = shap_v1["ordertable-gift_item"].astype("category", ordered=True,categories=ordertable_gift_item_categories).cat.codes
#shap_v1["newtable-hasdiscount"] = shap_v1["newtable-hasdiscount"].astype("category", ordered=True, categories=newtable_hasdiscount_categories).cat.codes
shap_v1["user_level"] = shap_v1["user_level"].astype("category", ordered=True,categories=userlevel_categories).cat.codes
shap_v1["education"] = shap_v1["education"].astype("category", ordered=True,categories=education_categories).cat.codes
shap_v1["city_level"] = shap_v1["city_level"].astype("category", ordered=True,categories=citylevel_categories).cat.codes
shap_v1["purchase_power"] = shap_v1["purchase_power"].astype("category", ordered=True,categories=purchastingpower_categories).cat.codes

geneder_categories = ['U', 'F', 'M']
channel_categories = ['app', 'wechat', 'pc', 'mobile', 'others']
marital_categories = ['U', 'M', 'S']

shap_v1["gender"] = shap_v1["gender"].astype("category", ordered=True,categories=geneder_categories).cat.codes

shap_v1["channel"] = shap_v1["channel"].astype("category", ordered=True,categories=channel_categories).cat.codes

shap_v1["marital_status"] = shap_v1["marital_status"].astype("category", ordered=True,categories=marital_categories).cat.codes


shap_v1['user_plus'] = shap_v1['user_plus'].astype('int8')
shap_v1['day_of_week'] = shap_v1['day_of_week'].astype('int8')
shap_v1['hour_of_day'] = shap_v1['hour_of_day'].astype('int8')
shap_v1['product_type'] = shap_v1['product_type'].astype('int8')
shap_v1['promise'] = pd.to_numeric(shap_v1['promise'], errors = 'coerce')
shap_v1['number_of_gifts'] = shap_v1['number_of_gifts'].astype('float64')

y = shap_v1['class']
X = shap_v1.drop(columns=["class"])

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class':4,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.04,
   'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'seed': 8,
}

random_state = 5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)

model = lgb.train(params, d_train, valid_sets=[d_test], early_stopping_rounds=50)
y_pred = model.predict(X_test)

filename = 'finalized_approach1_classification.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))


#argmax() method 
y_pred_1 = [np.argmax(line) for line in y_pred]

#using precision score for error metrics
precision_score(y_pred_1,y_test,average=None).mean()

gnb = GaussianNB()
y_pred_naive = gnb.fit(X_train, y_train).predict(X_test)

neigh = KNeighborsClassifier(n_neighbors=3)
knn_fit = neigh.fit(X_train, y_train)
y_pred_knn = knn_fit.predict(X_test)

#using precision score for error metrics
precision_score(y_pred_naive,y_test,average=None).mean()

#using precision score for error metrics
precision_score(y_pred_knn,y_test,average=None).mean()

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

shap.summary_plot(shap_values, features=X_test, class_inds=[0])
shap.summary_plot(shap_values, features=X_test, class_inds=[1])
shap.summary_plot(shap_values, features=X_test, class_inds=[2])
shap.summary_plot(shap_values, features=X_test, class_inds=[3])

shap.summary_plot(shap_values[0], features=X_test)
shap.summary_plot(shap_values[1], features=X_test)
shap.summary_plot(shap_values[2], features=X_test)
shap.summary_plot(shap_values[3], features=X_test)

shap.dependence_plot("percentage_quantity_discount_per_unit", shap_values[0], X_test, interaction_index=None)
shap.dependence_plot("percentage_quantity_discount_per_unit", shap_values[0], X_test)
shap.dependence_plot("percentage_quantity_discount_per_unit", shap_values[1], X_test, interaction_index=None)
shap.dependence_plot("percentage_quantity_discount_per_unit", shap_values[1], X_test)
shap.dependence_plot("percentage_quantity_discount_per_unit", shap_values[2], X_test, interaction_index=None)
shap.dependence_plot("percentage_quantity_discount_per_unit", shap_values[2], X_test)
shap.dependence_plot("percentage_quantity_discount_per_unit", shap_values[3], X_test, interaction_index=None)
shap.dependence_plot("percentage_quantity_discount_per_unit", shap_values[3], X_test)

X_test_disp = X_test.copy()

X_test_disp['age'] = X_test_disp['age'].map(dict(enumerate(age_categories)))
#X_test_disp['ordertable-gift_item'] = X_test_disp['ordertable-gift_item'].map(dict(enumerate(ordertable_gift_item_categories)))
#X_test_disp['newtable-hasdiscount'] = X_test_disp['newtable-hasdiscount'].map(dict(enumerate(newtable_hasdiscount_categories)))
X_test_disp['user_level'] = X_test_disp['user_level'].map(dict(enumerate(userlevel_categories)))
X_test_disp['education'] = X_test_disp['education'].map(dict(enumerate(education_categories)))
X_test_disp['city_level'] = X_test_disp['city_level'].map(dict(enumerate(citylevel_categories)))
X_test_disp['purchase_power'] = X_test_disp['purchase_power'].map(dict(enumerate(purchastingpower_categories)))


geneder_categories = ['U', 'F', 'M']
channel_categories = ['app', 'wechat', 'pc', 'mobile', 'others']
marital_categories = ['U', 'M', 'S']


X_test_disp['gender'] = X_test_disp['gender'].map(dict(enumerate(geneder_categories)))
X_test_disp['channel'] = X_test_disp['channel'].map(dict(enumerate(channel_categories)))
X_test_disp['marital_status'] = X_test_disp['marital_status'].map(dict(enumerate(marital_categories)))

# load JS visualization code to notebook
shap.initjs() 

plt.clf()
shap.force_plot(explainer.expected_value[1], shap_values[1][16,:], X_test_disp.iloc[16,:], link="logit", matplotlib=True,show=False,  figsize=(20,4))
plt.savefig("class2_classification_latest.png",bbox_inches='tight')

shap.decision_plot(explainer.expected_value[1], shap_values[1][16,:], X_test_disp.iloc[16,:], link="logit")

class0 = ['ordertable-original_unit_price', 'percentage_direct_discount_per_unit', 'product_type',
         'percentage_quantity_discount_per_unit', 'promise', 'percentage_bundle_discount_per_unit', 'percentage_coupon_discount_per_unit']
         
class1 = ['ordertable-original_unit_price', 'percentage_direct_discount_per_unit', 'product_type',
         'percentage_quantity_discount_per_unit', 'promise', 'number_of_gifts', 'day_of_week']
         
class2 = ['ordertable-original_unit_price', 'percentage_direct_discount_per_unit', 'product_type',
         'channel', 'percentage_bundle_discount_per_unit', 'percentage_coupon_discount_per_unit', 'percentage_quantity_discount_per_unit']
         
class3 = ['ordertable-original_unit_price', 'percentage_direct_discount_per_unit', 'product_type',
         'promise', 'education', 'percentage_coupon_discount_per_unit', 'percentage_quantity_discount_per_unit']

plt.clf()

for i in class0: 
    interaction_name = i
    with_interaction = "class1_" + interaction_name + "_with_interaction" + ".png"
    shap.dependence_plot(interaction_name, shap_values[1], X_test,show=False)
    plt1.savefig(with_interaction,bbox_inches='tight') 

for i in class0: 
    interaction_name = i
    with_interaction = "class1_" + interaction_name + "_without_interaction" + ".png"
    shap.dependence_plot(interaction_name, shap_values[1], X_test,show=False, interaction_index=None)
    plt1.savefig(with_interaction,bbox_inches='tight') 


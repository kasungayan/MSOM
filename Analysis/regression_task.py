import pandas as pd
import numpy as np
import plotly
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn import linear_model

np.random.seed(123)

shap_v1 = pd.read_feather('Preprocessed_data/click_data_shap_final_version.ft')

shap_v1['percentage_direct_discount_per_unit'] = shap_v1['ordertable-direct_discount_per_unit']/shap_v1['ordertable-original_unit_price']
shap_v1['percentage_quantity_discount_per_unit'] = shap_v1['ordertable-quantity_discount_per_unit']/shap_v1['ordertable-original_unit_price']
shap_v1['percentage_bundle_discount_per_unit'] = shap_v1['ordertable-bundle_discount_per_unit']/shap_v1['ordertable-original_unit_price']
shap_v1['percentage_coupon_discount_per_unit'] = shap_v1['ordertable-coupon_discount_per_unit']/shap_v1['ordertable-original_unit_price']

shap_v1 = shap_v1.drop(columns = ['ordertable-direct_discount_per_unit', 'ordertable-quantity_discount_per_unit', 'ordertable-bundle_discount_per_unit','ordertable-coupon_discount_per_unit'])

shap_v1.rename(columns = {'usertable-age':'age', 
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
                         'skutable-attribute1':'product_attribute1',
                          'skutable-attribute2':'product_attribute2',
                          'usertable-plus':'user_plus',
                          'ordertable-number_of_gifts':'number_of_gifts'
                         }, inplace = True)
                         
shap_v1['promise'] =shap_v1['promise'].replace({'nilorder': np.nan})

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
shap_v1['product_type'] = shap_v1['product_type'].astype('int8')
shap_v1['promise'] = pd.to_numeric(shap_v1['promise'], errors = 'coerce')

y = shap_v1['quantity']
X = shap_v1.drop(columns=["quantity"])

random_state = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)


#build the lightgbm model
params = {
          "objective" : "regression",
          "metric" :"rmse",
          "force_row_wise" : True,
          "learning_rate" : 0.075,
          "sub_row" : 0.75,
          "bagging_freq" : 1,
          "lambda_l2" : 0.1,
          "metric": ["rmse"],
          'verbosity': 1,
          'num_iterations' : 1200,
          'num_leaves': 128,
          "min_data_in_leaf": 100,
         }
         
         
model = lgb.train(params, d_train, valid_sets=[d_test], early_stopping_rounds=50)

explainer = shap.TreeExplainer(model)
X_test_disp = X_test.copy()

X_test_disp['age'] = X_test_disp['age'].map(dict(enumerate(age_categories)))
#X_test_disp['ordertable-gift_item'] = X_test_disp['ordertable-gift_item'].map(dict(enumerate(ordertable_gift_item_categories)))
#X_test_disp['newtable-hasdiscount'] = X_test_disp['newtable-hasdiscount'].map(dict(enumerate(newtable_hasdiscount_categories)))
X_test_disp['user_level'] = X_test_disp['user_level'].map(dict(enumerate(userlevel_categories)))
X_test_disp['education'] = X_test_disp['education'].map(dict(enumerate(education_categories)))
X_test_disp['city_level'] = X_test_disp['city_level'].map(dict(enumerate(citylevel_categories)))
X_test_disp['purchase_power'] = X_test_disp['purchase_power'].map(dict(enumerate(purchastingpower_categories)))

eneder_categories = ['U', 'F', 'M']
channel_categories = ['app', 'wechat', 'pc', 'mobile', 'others']
marital_categories = ['U', 'M', 'S']


X_test_disp['gender'] = X_test_disp['gender'].map(dict(enumerate(geneder_categories)))
X_test_disp['channel'] = X_test_disp['channel'].map(dict(enumerate(channel_categories)))
X_test_disp['marital_status'] = X_test_disp['marital_status'].map(dict(enumerate(marital_categories)))

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar", max_display = 21)

shap.summary_plot(shap_values,X_test,  max_display = 21)

predictions = model.predict(X_test)
predictions[predictions < 0] = 0

actual = y_test

print("RMSE Error LightGBM:{}".format(np.sqrt(metrics.mean_squared_error(actual,predictions))))


X_test.rename(columns = {'ordertable-original_unit_price':'original_unit_price'},inplace = True)

shap.dependence_plot("promise", shap_values, X_test, interaction_index=None)
shap.dependence_plot("promise", shap_values, X_test)

shap.dependence_plot("percentage_direct_discount_per_unit", shap_values, X_test)
shap.dependence_plot("percentage_direct_discount_per_unit", shap_values, X_test, interaction_index=None)


shap.dependence_plot("original_unit_price", shap_values, X_test)
shap.dependence_plot("original_unit_price", shap_values, X_test, interaction_index=None)

shap.dependence_plot("percentage_quantity_discount_per_unit", shap_values, X_test)
shap.dependence_plot("percentage_quantity_discount_per_unit", shap_values, X_test, interaction_index=None)

shap.dependence_plot("percentage_bundle_discount_per_unit", shap_values, X_test)
shap.dependence_plot("percentage_bundle_discount_per_unit", shap_values, X_test, interaction_index=None)

shap.dependence_plot("percentage_coupon_discount_per_unit", shap_values, X_test)
shap.dependence_plot("percentage_coupon_discount_per_unit", shap_values, X_test, interaction_index=None)

shap.dependence_plot("user_level", shap_values, X_test)
shap.dependence_plot("user_level", shap_values, X_test,interaction_index=None)

shap.dependence_plot("education", shap_values, X_test)
shap.dependence_plot("education", shap_values, X_test, interaction_index=None)

# load JS visualization code to notebook
shap.initjs()

plt.clf()
shap.force_plot(explainer.expected_value, shap_values[1594,:], X_test_disp.iloc[1594,:], matplotlib=True, show=False, figsize=(20,4))
plt.savefig("prediction1.png",bbox_inches='tight')

plt.clf()
shap.force_plot(explainer.expected_value, shap_values[1594,:], X_test_disp.iloc[1594,:], matplotlib=True, show=False, figsize=(20,4))
plt.savefig("prediction1.eps",bbox_inches='tight', format='eps')

shap.decision_plot(explainer.expected_value, shap_values[1594,:], X_test_disp.iloc[1594,:])








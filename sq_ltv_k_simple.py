
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import sys
from scipy import optimize

#===============================Simple LTV projection==================================
#This file contains a simple LTV calculation excercise with an
# assumption that sales to eaach user reached saturation quite fast 
# (we do not keep selling to the existing users at the same pace as initially)
# and also that lifetime for our user is estimated as 180 days - after 180 days the user 'dies'
# also K-values calculated to illustrate the ending point for our cumulative sales
#=======================================================================================


def get_cum_sum(data, date):
    return data['sum'].where(data['pay_date']<=date).sum()

def get_users_utd(data, date):
    return data['user'].where(data['install_date'] <=date).count()

def preprocess(data):
    data['install_date'] = pd.to_datetime(data['install_date'],dayfirst = True)
    data['pay_date'] = pd.to_datetime(data['pay_date'], dayfirst = True)
    data = data.sort_values('pay_date')
    data['cum_sum'] = data['pay_date'].map(lambda x: get_cum_sum(data, x))
    data['users_n_utd'] = data['pay_date'].map(lambda x: get_users_utd(data, x))
    data['ltv'] = data['cum_sum']/data['users_n_utd'].astype(float)
    data['day'] = pd.to_timedelta(data['pay_date'] - data['install_date'].min()).dt.days + 1
    data['day'] = data['day'].astype(int)
    return data

def ltv_func(param, c):
    result = c[0] + c[1]*np.log(param)
    return result

#=========reading and processing data==============
try:
    raw_data = pd.read_csv('sq_data.csv')
    ltv_data = preprocess(raw_data)
    x = ltv_data['day']
    y = ltv_data['ltv']

    #model LTV with log curve fit 
    coefs, cov = optimize.curve_fit(lambda p,a,b: a+b*np.log(p),  x,  y)
    ltv_90_180 = ltv_func([90., 180.], coefs)
    LTV_90 = round(ltv_90_180[0],2)
    LTV_180 = round(ltv_90_180[1],2)
    print ("LTV by day 90: " + str(LTV_90))
    print ("LTV by day 180: " + str(LTV_180))

    #calculate Ks
    k_days = [1, 3, 7, 30]
    for k in k_days:
        k_revenue = ltv_data['cum_sum'].loc[ltv_data['day'] == k].values[0] 
        coeff = LTV_180/k_revenue
        print ("K" + str(k) + ": " + str(round(coeff, 2)))
except:
    e = sys.exc_info()[0]
    print ('Whatever happens here, please, stay happy! :)')
    print ( "<p>Error: %s</p>" % e )


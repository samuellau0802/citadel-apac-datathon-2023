import pandas as pd
import numpy as np
import datetime
from scipy import stats
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/lsy/Downloads/APAC_2023_Datasets/APAC_2023_Datasets/Crashes/crash_info_general.csv')
data2 = data.loc[:, lambda d: d.columns.str.contains('COUNT') | d.columns.str.contains('TM')
                              |d.columns.str.contains('CRN') | d.columns.str.contains('TIME')]
data2 = data2.set_index('CRN')

time_ana = data2.loc[:,lambda d: d.columns.str.contains('TM')| d.columns.str.contains('TIME')]
time_ana.loc[lambda d:d['ARRIVAL_TM'] == d['TIME_OF_DAY'],'DISPATCH_TM'] = time_ana.loc[lambda d:d['ARRIVAL_TM'] == d['TIME_OF_DAY'],'ARRIVAL_TM']
time_pro = time_ana.loc[(time_ana != 9999).all(axis=1)]
time_pro = time_pro.dropna(axis = 0,how = 'any')
columns = time_pro.columns.to_list()
for i in columns:
    time_pro[i] = time_pro[i].apply(lambda _: str(int(_)).rjust(4,'0'))
    time_pro[i] = time_pro[i].apply(lambda _: datetime.datetime.strptime(_,'%H%M'))
time_pro['Temp_RES_TM_DAY'] = time_pro.loc[:,'ARRIVAL_TM'] - time_pro.loc[:,'DISPATCH_TM']
time_pro['RES_TM_DAY'] = time_pro['Temp_RES_TM_DAY'].apply(lambda d: d.days)
time_pro['RES_TM_min'] = time_pro['Temp_RES_TM_DAY'].apply(lambda d: d.seconds/60)

time_pro['Temp_TOT_RES_TM_DAY'] = time_pro.loc[:,'ARRIVAL_TM'] - time_pro.loc[:,'TIME_OF_DAY']
time_pro['TOT_RES_TM_DAY'] = time_pro['Temp_TOT_RES_TM_DAY'].apply(lambda d: d.days)
time_pro['TOT_RES_TM_min'] = time_pro['Temp_TOT_RES_TM_DAY'].apply(lambda d: d.seconds/60)

time_pro['Temp_CALL_TM_DAY'] = time_pro.loc[:,'DISPATCH_TM'] - time_pro.loc[:,'TIME_OF_DAY']
time_pro['CALL_TM_DAY'] = time_pro['Temp_CALL_TM_DAY'].apply(lambda d: d.days)
time_pro['CALL_TM_min'] = time_pro['Temp_CALL_TM_DAY'].apply(lambda d: d.seconds/60)

# time interval
time_tot_res = time_pro[time_pro['TOT_RES_TM_DAY'] >= 0]['TOT_RES_TM_min']
time_res = time_pro.loc[time_pro['RES_TM_DAY'] >= 0, 'RES_TM_min']
time_call = time_pro.loc[time_pro['CALL_TM_DAY'] >= 0, 'CALL_TM_min']

# get rid of anormaly
u1,u2,u3 = time_tot_res.mean(),time_res.mean(),time_call.mean()
s1,s2,s3 = time_tot_res.std(),time_res.std(),time_call.std()
time_tot_res = pd.DataFrame(time_tot_res)
time_tot_res = time_tot_res.loc[time_tot_res['TOT_RES_TM_min'] < u1+s1,:]
time_res = pd.DataFrame(time_res)
time_res = time_res.loc[time_res['RES_TM_min'] < u2+s2,:]
time_call = pd.DataFrame(time_call)
time_call = time_call.loc[time_call['CALL_TM_min'] < u3+s3,:]

# time_tot_res.to_csv('total_rsponse_miniutes.csv')
# time_res.to_csv('rsponse_miniutes.csv')
# time_call.to_csv('call_miniutes.csv')

data_count = data2.loc[:,lambda d: d.columns.str.contains('COUNT')]
data_tot_res = pd.merge(time_tot_res,data_count,how = 'left',on = 'CRN')
data_tot_res = data_tot_res.dropna(axis = 0,how = 'any')

data_res = pd.merge(time_res,data_count,how = 'left',on = 'CRN')
data_res = data_res.dropna(axis = 0,how = 'any')

data_call = pd.merge(time_call,data_count,how = 'left',on = 'CRN')
data_call = data_call.dropna(axis = 0,how = 'any')

# correlation test

# #pearson correlation
# #step1. test whether data is normal distributed
# u1,u2,u3 = time_tot_res.mean()[0],time_res.mean()[0],time_call.mean()[0]
# s1,s2,s3 = time_tot_res.std()[0],time_res.std()[0],time_call.std()[0]
#
# print('TOT_RES normal test：\n',stats.kstest(time_tot_res, 'norm', (u1, s1)))
# print('RES normal test：\n',stats.kstest(time_res, 'norm', (u2, s2)))
# print('CALL normal test：\n',stats.kstest(time_call, 'norm', (u3, s3)))
#
# if stats.kstest(time_tot_res, 'norm', (u1, s1)).pvalue < 0.05:
#     print('TOT_RES not normal, pearson test cannot be applied')
#
# if stats.kstest(time_res, 'norm', (u2, s2)).pvalue < 0.05:
#     print('RES not normal, pearson test cannot be applied')
#
# if stats.kstest(time_call, 'norm', (u3, s3)).pvalue < 0.05:
#     print('CALL not normal, pearson test cannot be applied')


# Spearman correlation
result_tot = pd.DataFrame(data_tot_res.corr('spearman').iloc[:,0])
result_res = pd.DataFrame(data_res.corr('spearman').iloc[:,0])
result_call = pd.DataFrame(data_call.corr('spearman').iloc[:,0])

#p_value
p_tot = stats.spearmanr(data_tot_res).pvalue[:,0]
p_res = stats.spearmanr(data_res).pvalue[:,0]
p_call = stats.spearmanr(data_call).pvalue[:,0]

#add p_value to result
result_tot.columns = ['spearman_corr']
result_res.columns = ['spearman_corr']
result_call.columns = ['spearman_corr']

result_tot['p_value'] = p_tot
result_res['p_value'] = p_res
result_call['p_value'] = p_call

result_tot = result_tot.dropna(axis = 0,how = 'any')
result_res = result_res.dropna(axis = 0,how = 'any')
result_call = result_call.dropna(axis = 0,how = 'any')

# filter p_value <0.05
correlated_tot = result_tot.loc[result_tot['p_value'] < 0.005,:].sort_values('p_value')
correlated_res = result_res.loc[result_res['p_value'] < 0.005,:].sort_values('p_value')
correlated_call = result_call.loc[result_call['p_value'] < 0.005,:].sort_values('p_value')

plt.figure(1)
plt.hist(np.asarray(time_tot_res))
plt.title('total response time')

plt.figure(2)
plt.hist(np.asarray(time_res))
plt.title('Response time from depatch')

plt.figure(3)
plt.hist(np.asarray(time_call))
plt.title('Time between occurrence and dispatch')
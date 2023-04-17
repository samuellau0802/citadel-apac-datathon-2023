import pandas as pd
import numpy as np
import datetime
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

data = pd.read_csv('APAC_2023_Datasets/Crashes/crash_info_general.csv')
data2 = data.loc[:, lambda d: d.columns.str.contains('COUNT') | d.columns.str.contains('TM')
                              |d.columns.str.contains('CRN') | d.columns.str.contains('TIME')
                              |d.columns.str.contains('CRASH_') ]
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

M_Y = data2.loc[:,['CRASH_MONTH','CRASH_YEAR']]
time_tot_m_y = pd.merge(time_tot_res,M_Y,how = 'left',on = 'CRN')
time_res_m_y = pd.merge(time_res,M_Y,how = 'left',on = 'CRN')
time_call_m_y = pd.merge(time_call,M_Y,how = 'left',on = 'CRN')
# time_tot_m_y = time_tot_m_y.groupby(['CRASH_YEAR','CRASH_MONTH'])['TOT_RES_TM_min'].mean()
# time_res_m_y = time_res_m_y.groupby(['CRASH_YEAR','CRASH_MONTH'])['RES_TM_min'].mean()
# time_call_m_y = time_call_m_y.groupby(['CRASH_YEAR','CRASH_MONTH'])['CALL_TM_min'].mean()


time_tot_y = time_tot_m_y.groupby(['CRASH_YEAR'])['TOT_RES_TM_min'].mean()
time_res_y = time_res_m_y.groupby(['CRASH_YEAR'])['RES_TM_min'].mean()
time_call_y = time_call_m_y.groupby(['CRASH_YEAR'])['CALL_TM_min'].mean()


year = time_tot_y.index.to_list()

xx = np.arange(len(year))
year = ['2010','2012','2014','2016','2018','2020']
f,ax = plt.subplots(2,3)
ax[0][0].plot(xx,np.asarray(time_tot_y))
ax[0][0].set_title('TOT_RES_TIME')
ax[0][0].set_xlabel('Year')
ax[0][0].set_xticks([0,2,4,6,8,10])
ax[0][0].set_xticklabels(year)
ax[0][0].set_ylabel('Time(min)')

ax[0][1].plot(xx,np.asarray(time_res_y))
ax[0][1].set_title('RES_TIME')
ax[0][1].set_xlabel('Year')
ax[0][1].set_xticks([0,2,4,6,8,10])
ax[0][1].set_xticklabels(year)

ax[0][2].plot(xx,np.asarray(time_call_y))
ax[0][2].set_title('CALL_TIME')
ax[0][2].set_xlabel('Year')
ax[0][2].set_xticks([0,2,4,6,8,10])
ax[0][2].set_xticklabels(year)
#
time_tot_m = time_tot_m_y.groupby(['CRASH_MONTH'])['TOT_RES_TM_min'].mean()
time_res_m = time_res_m_y.groupby(['CRASH_MONTH'])['RES_TM_min'].mean()
time_call_m = time_call_m_y.groupby(['CRASH_MONTH'])['CALL_TM_min'].mean()


month = time_tot_m.index.to_list()

xx = np.arange(len(month))
month = ['1','3','5','7','9','11']
ax[1][0].plot(xx,np.asarray(time_tot_m))
ax[1][0].set_title('TOT_RES_TIME')
ax[1][0].set_xlabel('Month')
ax[1][0].set_xticks([0,2,4,6,8,10])
ax[1][0].set_xticklabels(month)
ax[1][0].set_ylabel('Time(min)')

ax[1][1].plot(xx,np.asarray(time_res_m))
ax[1][1].set_title('RES_TIME')
ax[1][1].set_xlabel('Month')
ax[1][1].set_xticks([0,2,4,6,8,10])
ax[1][1].set_xticklabels(month)


ax[1][2].plot(xx,np.asarray(time_call_m))
ax[1][2].set_title('CALL_TIME')
ax[1][2].set_xlabel('Month')
ax[1][2].set_xticks([0,2,4,6,8,10])
ax[1][2].set_xticklabels(month)
plt.show()

#group by month and year
tot_my = time_tot_m_y.groupby(['CRASH_YEAR','CRASH_MONTH'])['TOT_RES_TM_min'].mean()

res_my = time_res_m_y.groupby(['CRASH_YEAR','CRASH_MONTH'])['RES_TM_min'].mean()
call_my = time_call_m_y.groupby(['CRASH_YEAR','CRASH_MONTH'])['CALL_TM_min'].mean()
#
decompose_result_tot = seasonal_decompose(tot_my, period=10,extrapolate_trend='freq')
#
f1,ax1 = plt.subplots(3,1)
xx = np.arange(len(np.asarray(decompose_result_tot.trend)))
ax1[0].plot(np.asarray(decompose_result_tot.trend))
ax1[0].set_title('Trend')
ax1[1].plot(np.asarray(decompose_result_tot.seasonal))
ax1[1].set_title('Seasonality')
ax1[2].plot(np.asarray(decompose_result_tot.resid))
ax1[2].set_title('Residual')
plt.show()
# decompose_result_tot.plot()

# plot_acf(tot_my, lags=12)
# plot_acf(res_my,lags = 12)
# plot_acf(call_my,lags = 12)


plt.figure(2)
plt.plot(np.asarray(tot_my))
plt.title('Total Response Time')
plt.xlabel('Date')
plt.ylabel('time(min)')
plt.xticks(np.arange(0,144,12),['2010/01','2011/01','2012/01','2013/01',
                 '2014/01','2015/01','2016/01','2017/01','2018/01','2019/01','2020/01','2021/01'],rotation = 40)
plt.show()


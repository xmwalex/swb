# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:54:36 2018

@author: ximing

收盘的时候，看：
如果ma20>+8%, 收盘涨+0.1%, 收盘就做空。 次日收盘cover。
QQQ如果昨天收盘比ma10 不低于-12%， 然后昨天收盘比开盘上涨超过0.5%, 收盘做多,pre limit sell 1.1%. 如果开盘7:30跌幅超过0.4%, 7:30做空.收盘cover,否则hold 到收盘。
如果前一天创了新高，夜里挂1.1% limit sell, or 开盘sell并且做空,收盘cover. 但如果开盘跌幅超过0.2%，那么不要做空，hold 到收盘。
做空的场合，如果急跌之后，不要动，之后必然有个反弹，等反弹了横盘了在sell并做空。


晚上：
挂QQQ+1.1%的limit sell过夜，如果已经卖掉，开盘后跌了超过1.5%, buy back

早上6：30 看：
如果高开0.8% or 低开-1.8%, hold 到收盘
如果开盘低于+0.05%,挂+1.17% limit sell,如果10点前能卖掉就卖，不能就撤单等收盘
如果开盘高于+0.05%,挂+0.04%的sell。如果7点以前能卖掉，等7点钟如果低于+0.05%,挂-0.1% limit buy进去等收盘。否则一直挂着到收盘

七点钟跌0.2%的策略没有效！不要做！

测试期 5月7号


上面的策略太overfit了


新策略：
如果ma20>+8%, 收盘涨+0.1%, 收盘就做空。 次日收盘cover。
如果盘前或者高开超过1.1%,抛出。 如果之后当天比开盘价低1.5%，抄底
如果开盘低于2%，hold到收盘
如果开盘低于0.1%，之后如果当天在13点前涨1.1%，抛出，否则hold到收盘
否则的话，用比开盘价搞0.04%的bar卖出。卖不掉就hold 到收盘。 


"""

qqq = pd.read_csv('QQQtoday.csv')
ma20 = qqq.loc[len(qqq)-1,'Adj Close']/np.mean(qqq.loc[len(qqq)-20:]['Adj Close'])
ma10 = qqq.loc[len(qqq)-1,'Adj Close']/np.mean(qqq.loc[len(qqq)-10:]['Adj Close'])

if ma20>1.08 and qqq.loc[len(qqq)-1,'Adj Close']>1.001*qqq.loc[len(qqq)-2,'Adj Close']:
    print('short at close, cover next close')
if ma10>.88 and qqq.loc[len(qqq)-1,'Close']>1.007*qqq.loc[len(qqq)-1,'Open']:
    print('long at close. sell and short at next 10am. if after short, down 3%, cover. otherwise cover at close')
print('sell limit at 1.2% of QQQ, 3.6% of TQQQ before 10am. if Sold, buy if down 1.5% of 10am open')
print('hold to close if 10am up 0.7% or down 0.5%')

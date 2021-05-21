import numpy as np
import pandas as pd
from Fin import *
from scipy import stats
import statsmodels.api as sm
    
def moveone(df):
    arlist = [0]
    for i in range(len(df['AR'])-1):
        arlist.append(df['AR'].iloc[i])
    return arlist
    
def MakePortbyCriteria(df, Name ):
    dfs = []
    Criterias = sorted(df[Name].unique().tolist())
    for i in Criterias:
        now = df[ df[Name] == i ]
        now = Port(now)
        dfs.append(now)
    return dfs

def totalComNum(dfs, criteria = [0,1,2,3,4]):
    num = 0 
    for i in criteria:
        num += dfs[i].counts.sum()
    return num

class Port():
    """
    검증을 위한 포트폴리오 만들기
    AR, CR, Return만들어야함
    """
    def __init__(self,df):
        self.maindf = df.fillna(0)
        self.up10value = self.Make10port()
        self.Up10com = self.up10value['deltaprice']
        self.CRreturnlist = self.maindf['deltaprice']
        self.CR = self.CRreturnlist.mean()
        self.ARreturnlist = self.maindf['deltaprice'] - self.Up10com.mean()
        self.maindf['AR'] = self.ARreturnlist
        self.AR = self.ARreturnlist.mean()
        self.counts = self.maindf.count()['Name']
        
    def Make10port(self):
        MarketValue10 = self.maindf['시가총액'].quantile(0.9)
        Port10 = self.maindf[self.maindf['시가총액']>=MarketValue10]
        return Port10
    
    def ConsenPEIS(self):
        self.PEIS = MakePortbyPEIS(self.maindf) # 여기서 이미 컨센별로 포폴이 나눠져있어서 maindf는 컨센별임 
    
    def Maketable8(self):
        """
        기업,티커별로 나누고 거기서 RET만들어서 넣어줘야겠음
        --> 자꾸 에러나니깐 그냥 AR을 한층들어 올리고 티커 기업이름 별로 묶어서 한층 떨구자 
        """
        self.maindf['RET'] = moveone(self.maindf)
        # drop 
        tempdf = self.maindf.set_index(['Symbol','Name'])
        indexs = tempdf.index.unique()
        Comlist = []
        for ticker, name in indexs:
            nowcom = tempdf.loc[(ticker),:].iloc[1:]
            Comlist.append(nowcom)
        self.table8 = pd.concat(Comlist) 
    

    
    def HEDGE(self,i): 
        self.maindf['HEDGE'] = i
        
    
def MakePortbyPEIS(df):
    dfs = []
    quans = []
    for i in [0,0.2,0.4,0.6,0.8,1]:
        quans.append(df['valPEIS'].quantile(i))
    
    for i in range(5):
        temp = df[ (df['valPEIS'] >= quans[i]) & ( df['valPEIS'] < quans[i+1] ) ]
        if ( i == 4 ):
            temp = pd.concat([ temp, df[df['valPEIS'] == quans[5]]     ])
        temp = Port(temp)
        dfs.append(temp)
        
    return dfs
        
def hedge1(df):
    df['HEDGE'] = 1
    
def MakeTable6form(dfs):
    totalcompany = []
    Earnincrease = []
    for df in dfs :
        totalcompany.append(df.counts.sum())
        Earnincrease.append( df.maindf['deltaearn'].sum() )
    Earnin = np.array(Earnincrease)
    totalcom = np.array(totalcompany)
    outtable = pd.DataFrame( [Earnin, totalcom - Earnin, Earnin/totalcompany],
                           columns = [5,4,3,2,1],index = ['Earning increases', 'Earning decrease', 'Percentage of increse'])
    return round(outtable.T,2)        
        
def MultiLinear(buys, sell = pd.DataFrame()):
    nsell = len(sell)
    buy = pd.concat([buys])
    if nsell != 0:
        hedge1(buy)        
        sell['AR'] = sell['AR'] * -1
        All = pd.concat([buy,sell])
    else:
        All = buy
    nAll = len(All)
    print("N : ", nAll)
    if nsell != 0:
        X = All[['HEDGE','PBR','RET','PER','ACC','BETA']]
    else :
        X = All[['PBR','RET','PER','ACC','BETA']]
    
    y = All['AR']
    X1 = sm.add_constant(X,has_constant = 'add')
    model1 = sm.OLS(y,X1)
    result1 = model1.fit()
    return result1.summary()

def ttest(testlist, ConsenPorts):
    for i in testlist:
        x, y = ConsenPorts[i].PEIS[0].ARreturnlist, ConsenPorts[i].PEIS[4].ARreturnlist
        _stat,_pvalue = stats.levene(x,y)
        if _pvalue < 0.05:
            equal_Var = False
        else :
            equal_Var = True
        statistic , pvalue = stats.ttest_ind(x,y, equal_var= equal_Var)
        print("\nstatistic : %d , pvalue : %.7f\n" % (statistic, pvalue))        
        
def ConsenPEISTable7(ConsenPorts):
    TotalConsenPEIS = [0,0,0,0,0]
    for dfs in ConsenPorts:
        name = dfs.maindf['consen_cut'].unique().tolist()
        temp2 = []
        for df in dfs.PEIS:
            data = np.array([df.counts.sum(), df.CR, df.AR ])
            data[np.isnan(data)] = 0        
            indexs = [name*3, ['N','CR (%)','AR (%)']  ]
            temp1 = pd.DataFrame(data,index = indexs)
            temp2.append(temp1)
        temp2 = pd.concat(temp2,axis=1) # 이게 한PEIS에 대해서 일자로 합친 것
        TotalConsenPEIS[name[0]] = temp2
    return pd.concat(TotalConsenPEIS)        

def Maketable8(clfs):
    for clf in clfs:
        clf.ConsenPEIS()
        clf.Maketable8()
        
def AllTable7(ConsenPorts):
    Table7 = [0,0,0,0,0]
    for df in ConsenPorts:
        data = np.array([df.counts.sum(), round(df.CR,2), round(df.AR,2) ])
        name = df.maindf['consen_cut'].unique().tolist()
        indexs = [name*3, ['N','CR (%)','AR (%)']  ]
        temp1 = pd.DataFrame(data,index = indexs)
        Table7[name[0]] = temp1
    return pd.concat(Table7)        
        
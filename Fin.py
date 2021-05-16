import pandas as pd
import numpy as np
import math 

def MakeStockTableList(df):
    DfList = []
    for i in range( int(len(df) / 6 ) ):
        nowdf = pd.DataFrame(df.iloc[i:i+6,:])
        DfList.append(nowdf)
    return DfList

def Calminus(df):
    minuslist = [0]
    for i in range(len(df)-1):
        minuslist.append(df.iloc[i+1] - df.iloc[i]) 
    return minuslist

def Caldelta(df):
    deltalist = [0]
    for i in range(len(df)-1):
        deltalist.append( (df.iloc[i+1] - df.iloc[i]) / df.iloc[i]) 
    return deltalist

def Caldeltaprice(df):
    deltalist = []
    for i in range(1,len(df)):
        deltalist.append( (df[i+1] - df[i]) / df[i]) 
    deltalist.append(0)
    return deltalist

def CalGM(df):
    df['GM1'] = df["매출총이익(천원)"] # GM 구성요소 1매출총이익 성장률
    df['deltaGM1'] = Caldelta(df['GM1'])
    df['deltasale'] = Caldelta(df['매출액(천원)'])
    df['deltaGM'] = df['deltaGM1'] - df['deltasale']
    return df

def CalSGA(df):
    # 이부분 논문 정확히 구현 불가능 교수님한테 물어보기 매출원가는 ok 영업비용, 일반관리비가 항목이 애매함
    df['SGA'] = df["판매비와관리비(천원)"] # 판관비만 SGA에 해당한다 매출원가는 COGS cost of goods sold로 분류됨
    df['SGAp'] = df['SGA'] / df['매출액(천원)']
    df['deltaSGA'] = Calminus(df['SGAp'])   
    
def Calhalf(df):
    deltalist = [0]
    for i in range(len(df)-1):
        deltalist.append( (df[i+1] + df[i]) / 2) 
    return deltalist

def CalGNOA(df):
    df['GNOA'] = Caldelta(df['NOA'])
    return df

def CalRNOA(df):
    df['NOA'] = df["보통주자본금(천원)"]+df["총부채(천원)"]+df["우선주자본금(천원)"]-df["현금및현금성자산(천원)"]-df["단기금융자산(천원)"]-df["단기매매금융자산(*)(천원)"] # 여기서 단기차입금이 아니라 단기금융상품임
    df['AVGNOA'] = Calhalf(df['NOA'])
    df['RNOA'] = df['영업이익(천원)']/df['AVGNOA']
    return df

def CalACC(df):
    df['deltaACC'] = ( df['영업이익(천원)'] - df['영업활동으로인한현금흐름(천원)'] ) / df['AVGNOA']
    return df

def CalforATO(u,d):
    datalist = [0]
    for i in range(len(u)-1):
        datalist.append( u[i+1] / d[i] )
    return datalist

def CalAto(df):
    df['ATO'] = CalforATO(df['매출액(천원)'] , df['총자산(천원)'])
    df['deltaATO'] = Calminus(df['ATO'])
    return df

def Earning(df):
    deltalist = []
    for i in range(len(df)-1):
        if df.iloc[i+1] > df.iloc[i]:
            deltalist.append(1)
        else:
            deltalist.append(0)
    deltalist.append(0)
    return deltalist


def setSGA(df):
    df['SGAdeltasale>0'] = 0
    df['SGAdeltasale<0'] = 0
    for i in range(len(df)):
        if( df['deltasale'].iloc[i] > 0):
            df['SGAdeltasale>0'].iloc[i] = df['deltaSGA'].iloc[i]
        elif( df['deltasale'].iloc[i] <= 0):
            df['SGAdeltasale<0'].iloc[i] = df['deltaSGA'].iloc[i]
    return df

def Cut20NullfillNA(df):
    """
    Null값 20넘어가는거는 삭제해줄 필요가 있음 그냥 아예 데이터가 없는 구간이 뽑힌것
    그보다 작은 NULL 값들은 모두 금융권 기업들임
    """
    df['CountNull'] = df.isnull().sum(axis=1)
    df = df.drop(df[ df['CountNull'] >= 20 ].index) # --> 여기를 20이상으로 해서 잡으면 댈 것 같은데?? 
    #df = df.drop(df[ df['CountNull'] == 23 ].index)
    df = df.fillna(0)
    return df


def CalPEIS(df):
    df['valPEIS'] = df['GNOAPEISpoint'] + df['deltaATOPEISpoint']+ df['deltaGMPEISpoint']+ df['deltaACCPEISpoint']+ df['RNOAPEISpoint']+ df['SGAdeltasale>0PEISpoint']+ df['SGAdeltasale<0PEISpoint']
    return df

def yearbaseDFs(df, years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020] ):
    dfs = []
    for i in years:
        now = df[ df['회계년'] == i ]
        dfs.append(now)
    return dfs

def namebaseDFs(df):
    dfs = []
    for i in range(int(len(df)/14)):
        j = i*14
        now = df.iloc[j:j+14,:]
        now = now.reset_index()
        now.drop('index',axis = 1, inplace = True)
        dfs.append(now)
    return dfs


def concatPEIS_STOCK(PEISdfs,namesymbol,Onetable):
    for i in range(int(len(namesymbol)/8)):
        for j in range(int(len(PEISdfs))):
            if( (namesymbol[i*8][0]==PEISdfs[j]['Name'][0]) & (namesymbol[i*8][1] == PEISdfs[j]['Symbol'][0]) ):    
                PEISdfs[j] = pd.concat([PEISdfs[j], Onetable.iloc[i*14:i*14+14,:].reset_index(drop=True)], axis=1)
                break ;
    return PEISdfs

def MakeLookGood(df, years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]):
    df = df.set_index(["Symbol Name", "Item Name "])
    df.drop('Symbol',axis=1,inplace=True)
    df.columns = years
    df.columns.names = ["year"]
    return df

def MakeOneLine(df, needstr, inputname):
    line = pd.Series(df.loc[(slice(None),needstr),:].fillna(0).stack(0).values,name=inputname)
    return line

def MakeOnetable(df, needstrs,inputnames):
    lines = []
    for needs, names in zip(needstrs,inputnames):
        line = MakeOneLine(df, needs, names)
        line.drop(0,inplace = True)
        line[len(line)+1] = 0
        lines.append(line)
    Onetable = pd.concat(lines,axis=1)
    return Onetable

def CompuMoreData(df):
    """
    기업별로 각각 PEIS 정보를 계산하기 때문에 섞일 수가 없는 듯 PEIS점수는 ㅇㅇ 
    deltaprice, 컨센 그런게 좀 걸리긴하겠네
    """
    ComName = set(df['Name'])
    df = df.set_index(["Name","회계년"])
    Comlist = []
    for name in ComName:
        nowcom = df.loc[(name,slice(None)), :]
        CalRNOA(nowcom)
        CalGNOA(nowcom)
        CalACC(nowcom)
        CalGM(nowcom)
        CalAto(nowcom)
        CalSGA(nowcom)
        setSGA(nowcom)
        nowcom['deltaearn'] = Earning(nowcom['영업이익(천원)'])
        Comlist.append(nowcom)
        
        nowcom
    ComputeMoreData = pd.concat(Comlist).reset_index()
    return ComputeMoreData

def Upper20Port(dfs, ColName):
    """
    음.. 데이터프레임 리스트를 받고, 거기서 for문으로 조건만족하는 것 고르고 
    바로 다음에 걔들 인덱스를 받아서 프레임 참조해서 수익률 평균치면 그 해 5월 사서 1년들어서 수익률임
    """
    ProfitList = []
    for i in dfs:
        now = i
        under20 = now[ColName].quantile(0.2)
        upper20 = now[ColName].quantile(0.8)
        condinow = now[now[ColName]>upper20]
        profit = np.mean(condinow['deltaprice'])
        ProfitList.append(profit)
    return ProfitList

def underplus(df):
    pointlist = []
    under20 = df.quantile([0.2,0.8]).iloc[0]
    upper20 = df.quantile([0.2,0.8]).iloc[1]
       
    for i in range(len(df)):
        if df.iloc[i] <= under20:
            pointlist.append(1)
        elif df.iloc[i] >= upper20:
            pointlist.append(-1)
        else :
            pointlist.append(0)
    pointlist = pd.DataFrame(pointlist, columns = [df.name+"PEISpoint"])
    return pointlist
        
def upperplus(df):
    pointlist = []
    under20 = df.quantile([0.2,0.8]).iloc[0]
    upper20 = df.quantile([0.2,0.8]).iloc[1]
       
    for i in range(len(df)):
        if df.iloc[i] <= under20:
            pointlist.append(-1)
        elif df.iloc[i] >= upper20:
            pointlist.append(1)
        else :
            pointlist.append(0)
    pointlist = pd.DataFrame(pointlist, columns = [df.name+"PEISpoint"])
    return pointlist


def underplus_SGA(df):
    pointlist = []
    under20 = np.percentile(df.unique(),20)
    upper20 = np.percentile(df.unique(),80)
        
    for i in range(len(df)):
        if df.iloc[i] <= under20:
            if df.iloc[i] == 0:
                pointlist.append(0)
            else:
                pointlist.append(1)
        elif df.iloc[i] >= upper20:
            if df.iloc[i] == 0:
                pointlist.append(0)
            else:
                pointlist.append(-1)
        else :
            pointlist.append(0)
    pointlist = pd.DataFrame(pointlist, columns = [df.name+"PEISpoint"])
    return pointlist

        
def upperplus_SGA(df):
    pointlist = []
    under20 = np.percentile(df.unique(),20)
    upper20 = np.percentile(df.unique(),80)
       
    for i in range(len(df)):
        if df.iloc[i] <= under20:
            if df.iloc[i] == 0:
                pointlist.append(0)
            else:
                pointlist.append(-1)
        elif df.iloc[i] >= upper20:
            if df.iloc[i] == 0:
                pointlist.append(0)
            else:
                pointlist.append(1)
        else :
            pointlist.append(0)
    pointlist = pd.DataFrame(pointlist, columns = [df.name+"PEISpoint"])
    return pointlist

def CalPEISpoints(dfs):
    YearDflist = []
    for df in dfs:
        df = df.reset_index()
        dfRNOA = underplus(df['RNOA'])
        dfGNOA = underplus(df['GNOA'])
        dfATO = upperplus(df['deltaATO'])
        dfGM = upperplus(df['deltaGM'])
        dfACC = underplus(df['deltaACC'])
        dfSGAupper = underplus_SGA(df['SGAdeltasale>0'])
        dfSGAunder = upperplus_SGA(df['SGAdeltasale<0'])
        now = pd.concat([df,dfGNOA, dfATO, dfGM, dfACC, dfRNOA, dfSGAupper, dfSGAunder],axis=1)
        YearDflist.append(now)
    NewDf = pd.concat(YearDflist)
    NewDf.drop(['index'], axis = 1, inplace = True)
    NewDf.reset_index(inplace = True)
    NewDf.drop(['index'], axis = 1, inplace = True)
    
    return NewDf

def returnspecindex(names):
    numlist = []
    for i in range(len(names)):
        if "스팩" in names[i]:
            numlist.append(i)
    return numlist

def CompuConsen(df):
    df['CompuConsen'] = 0
    Nowprice = df['Endprice'] # 지금 주가
    Targetprice = df['Targetprice'] # 목표주가 
    for i in range(len(df)):
        if (Targetprice[i] == -1) :
            df['CompuConsen'][i] = 0
        elif Nowprice[i]*1.4 <= Targetprice[i]:
            df['CompuConsen'][i] = 4
        elif ((Nowprice[i]*1.2 <= Targetprice[i]) & (Nowprice[i]*1.4 > Targetprice[i])) :
            df['CompuConsen'][i] = 3
        elif ((Nowprice[i]*0.8 <= Targetprice[i]) & (Nowprice[i]*1.2 > Targetprice[i])) :
            df['CompuConsen'][i] = 2
        elif (Nowprice[i]*0.8 > Targetprice[i]) :
            df['CompuConsen'][i] = 1
        else :
            df['CompuConsen'][i] = 0

def showTable5(df):
    df_mean = df.mean()
    df_std = np.sqrt(df.var())
    df_p20 = df.quantile(0.2)
    df_median = df.median()
    df_p80 = df.quantile(0.8)
    df_min = df.min()
    df_max = df.max()
    now = pd.concat([df_mean, df_std, df_p20, df_median, df_p80, df_min, df_max],axis=1)
    now.columns = ['Mean','Std','P20','Median','P80','Min','Max']
    Alldata = now.drop('consen_cut')
    print("Alldata count : ", len(df))
    df.set_index('consen_cut')
    
    consens = df['consen_cut'].sort_values().unique()
    listnum = [i for i in consens] *7
    listnum.sort()
    table5 = []
    for i in consens:
        df_mean = df[df['consen_cut'] == i ].mean()
        df_std = np.sqrt(df[df['consen_cut'] == i ].var())
        df_p20 = df[df['consen_cut'] == i ].quantile(0.2)
        df_median = df[df['consen_cut'] == i ].median()
        df_p80 = df[df['consen_cut'] == i ].quantile(0.8)
        df_min = df[df['consen_cut'] == i ].min()
        df_max = df[df['consen_cut'] == i ].max()
        now = pd.concat([df_mean, df_std, df_p20, df_median, df_p80, df_min, df_max],axis=1)
        print("consen : " +str(i) + " count : "  + str( df[df['consen_cut'] == i]['RNOA'].count() ))
        table5.append(now)
    table5 = pd.concat(table5)
    table5.drop('consen_cut',inplace = True)
    table5.columns = ['Mean','Std','P20','Median','P80','Min','Max']
    table5.index = [listnum, ['RNOA','deltaSGA','deltasale','deltaATO','GNOA','deltaACC','PEIS']*5]
    
    return Alldata, table5
    

def dropdata(df, colname, droplist):
    for i in droplist:
        dropindex = df[df[colname] == i].index
        df.drop(dropindex, inplace = True)
    df.reset_index(inplace = True)
    df.drop(['index'], axis = 1, inplace = True)

def dropdelist(df,delistcom):
    for i in delistcom:
        df.drop( df[df['Name'] == i].index, inplace =True )






























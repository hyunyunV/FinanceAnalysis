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
    for i in range(len(df)-1):
        deltalist.append( (df[i+1] - df[i]) / df[i]) 
    deltalist.append(0)
    return deltalist

def CalGM(df):
    df['SGA'] = df["매출원가(천원)"]+df["판매비와관리비(천원)"]
    df['deltaSGA'] = Calminus(df['SGA'])
    # 이부분 논문 정확히 구현 불가능 교수님한테 물어보기 매출원가는 ok 영업비용, 일반관리비가 항목이 애매함
    df['GM'] = df["매출총이익(천원)"]
    df['deltaGM1'] = Caldelta(df['GM'])
    df['deltasale'] = Caldelta(df['매출액(천원)'])
    df['deltaGM2'] = df['deltaGM1'] - df['deltasale']
    return df

def CalGNOA(df):
    df['GNOA'] = df['매출채권(천원)'] + df['재고자산(천원)'] - df['매입채무(천원)']
    df['deltaGNOA'] = Caldelta(df['GNOA'])
    return df

def CalRNOA(df):
    #df['NetAsset'] = df["총자산(천원)"]-df["현금및현금성자산(천원)"]-df["단기차입금(천원)"]-df["유동부채(천원)"]
    # 뭔지 모름
    df['NetAsset'] = df["보통주자본금(천원)"]+df["유동부채(천원)"]+df["비유동부채(천원)"]+df["우선주자본금(천원)"]-df["현금및현금성자산(천원)"]-df["단기차입금(천원)"]
    df['RNOA'] = df['영업이익(천원)']/df['NetAsset']
    return df

def CalACC(df):
    df['NetAsset'] = df["보통주자본금(천원)"]+df["유동부채(천원)"]+df["비유동부채(천원)"]+df["우선주자본금(천원)"]-df["현금및현금성자산(천원)"]-df["단기차입금(천원)"]
    df['deltaACC'] = ( df['영업이익(천원)'] - df['영업활동으로인한현금흐름(천원)'] ) / df['NetAsset']
    return df

def CalAto(df):
    df['ATO'] = df['매출액(천원)'] / df['총자산(천원)']
    df['deltaATO'] = Calminus(df['ATO'])
    return df

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
    df['CountNull'] = df.isnull().sum(axis=1)
    df = df.drop(df[ df['CountNull'] >= 20 ].index) # --> 여기를 20이상으로 해서 잡으면 댈 것 같은데?? 
    #df = df.drop(df[ df['CountNull'] == 23 ].index)
    df = df.fillna(0)
    df.drop(["Unnamed: 0"],axis = 1,inplace = True)
    df.drop(df[ (df['deltaprice']==99) | (df['deltaprice']==np.inf)].index, inplace = True)
    return df


def CalPEIS(df):
    df['valPEIS'] = df['GNOAPEISpoint'] + df['deltaATOPEISpoint']+ df['deltaGM2PEISpoint']+ df['deltaACCPEISpoint']+ df['RNOAPEISpoint']+ df['SGAdeltasale>0PEISpoint']+ df['SGAdeltasale<0PEISpoint']
    return df

def yearbaseDFs(df, years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020] ):
    dfs = []
    for i in years:
        now = df[ df['회계년'] == i ]
        dfs.append(now)
    dfs = dfs[1:]
    return dfs

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
        lines.append(line)
    Onetable = pd.concat(lines,axis=1)
    return Onetable

def CompuMoreData(df):
    ComName = set(df['Name'])
    df = df.set_index(["Name","회계년"])
    Comlist = []
    for name in ComName:
        nowcom = df.loc[(name,slice(None)), :]
        CalACC(nowcom)
        CalRNOA(nowcom)
        CalGNOA(nowcom)
        CalGM(nowcom)
        CalAto(nowcom)
        setSGA(nowcom)
        Comlist.append(nowcom)
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

def CalPEISpoints(dfs):
    YearDflist = []
    for df in dfs:
        df = df.reset_index()
        dfGNOA = underplus(df['GNOA'])
        dfATO = upperplus(df['deltaATO'])
        dfGM = upperplus(df['deltaGM2'])
        dfACC = underplus(df['deltaACC'])
        dfRNOA = underplus(df['RNOA'])
        dfSGAupper = underplus(df['SGAdeltasale>0'])
        dfSGAunder = upperplus(df['SGAdeltasale<0'])
        now = pd.concat([df,dfGNOA, dfATO, dfGM, dfACC, dfRNOA, dfSGAupper, dfSGAunder],axis=1)
        YearDflist.append(now)
    NewDf = pd.concat(YearDflist)
    return NewDf

def returnspecindex(names):
    numlist = []
    for i in range(len(names)):
        if "스팩" in names[i]:
            numlist.append(i)
    return numlist



def CompuConsen(df):
    df['CompuConsen'] = 0
    Nowprice = df['총자산(천원)'] # 지금 주가
    Targetprice = df['총자산(평균)(천원)'] # 목표주가 
    for i in range(len(df)):
        if Nowprice[i]*1.4 <= Targetprice[i]:
            df['CompuConsen'][i] = 5
        elif ((Nowprice[i]*1.2 <= Targetprice[i]) | (Nowprice[i]*1.4 > Targetprice[i])) :
            df['CompuConsen'][i] = 4
        elif ((Nowprice[i]*0.8 <= Targetprice[i]) | (Nowprice[i]*1.2 > Targetprice[i])) :
            df['CompuConsen'][i] = 3
        elif (Nowprice[i]*0.8 > Targetprice[i]) :
            df['CompuConsen'][i] = 1
        else :
            df['CompuConsen'][i] = 0
            


def showTable5(df):
    consens = df['consen_cut'].unique().sort_values()
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
        table5.append(now)
    table5 = pd.concat(table5)
    table5.drop('consen_cut',inplace = True)
    table5.columns = ['Mean','Std','P20','Median','P80','Min','Max']
    table5.index = [listnum, ['RNOA','deltaSGA','deltasale','deltaATO','GNOA','deltaACC','PEIS']*5]
    
    return table5



































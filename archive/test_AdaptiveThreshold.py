
def AdaptiveThreshold(totalRunID,table,ignoreUnMatch): # totalRunID start from 0
    class MyStruct():
        def __init__(self):
            self.field1 = 1
    cfg=MyStruct()
    cfg.session = int(table.loc[totalRunID,"ses"])
    cfg.run = int(table.loc[totalRunID,"run"])

    table = table.loc[:totalRunID]
    table_curr_ses=table[table['ses']==cfg.session]
    ThresholdList = list(table['threshold'])[:-1]
    SuccessList = list(table_curr_ses["progress"])[:-1] #成功列表
    
    # 如果现在是第1个session的第一个feedback training run
    # threshold=0.6
    if cfg.session == 1 and cfg.run == 1:
        threshold=0.7

    # 如果现在是第N个session的第一个feedback training run
    # threshold=前一天的最后一个threshold
    elif cfg.run == 1:
        try:
            threshold=float(ThresholdList[-1])
        except:
            threshold=0.6 #在极端情况下，我可能第二个session没有能够运行feedback session，就必须在第三个session的时候的第一个run才产生第一个threshold
    else:
        change = 0
        threshold=float(ThresholdList[-1])

        # 如果之前的1个run的进步是<=1
        # threshold=threshold-5%
        if SuccessList[-1] <= 1:
            change = change - 0.05

        # 如果之前的1个run的进步全部>=11
        # threshold=threshold+5%
        if SuccessList[-1] >= 11:
            change = change + 0.05

        if len(SuccessList)>=3:
            # 如果之前的3个run的进步全部<=3
            # threshold=threshold-5%
            if SuccessList[-1] <= 3 and SuccessList[-2] <= 3 and SuccessList[-3] <= 3:
                change = change - 0.05

            # 如果之前的3个run的进步全部>=9
            # threshold=threshold+5%
            elif SuccessList[-1] >= 9 and SuccessList[-2] >= 9 and SuccessList[-3] >= 9:
                change = change + 0.05

        if len(SuccessList)>=5:
            # 如果之前的5个run的进步全部<=5
            # threshold=threshold-5%
            if SuccessList[-1] <= 5 and SuccessList[-2] <= 5 and SuccessList[-3] <= 5 and SuccessList[-4] <= 5 and SuccessList[-5] <= 5:
                change = change - 0.05

            # 如果之前的5个run的进步全部>=7
            # threshold=threshold+5%
            elif SuccessList[-1] >= 7 and SuccessList[-2] >= 7 and SuccessList[-3] >= 7 and SuccessList[-4] >= 7 and SuccessList[-5] >= 7:
                change = change + 0.05
        # 如果之前的任意个run的进步全部【6】
        # threshold=threshold
        if SuccessList[-1] == 6:
            change = 0

        if change > 0.05:
            change = 0.05
        if change < -0.05:
            change = -0.05
        threshold = threshold + change

    # 不要越界
    if threshold>0.9:
        threshold=0.9
    if threshold<0.4:
        threshold=0.4

    # 如果这个run已经跑过了，给出这个error提醒。
    if len(table[(table['ses']==cfg.session) & (table['run']==cfg.run)])>1: #more robust than     # if ThresholdLog['session'].iloc[-1]==cfg.session and ThresholdLog['run'].iloc[-1]==cfg.run:
        print(f"this run exists!")
        raise Exception(f"this run exists!") 
    # print(f"threshold={threshold}")
    # print(f"float(table.loc[totalRunID,'threshold'])={float(table.loc[totalRunID,'threshold'])}")
    if float(round(threshold,2)) != float(table.loc[totalRunID,"threshold"]):
        if ignoreUnMatch<=0:
            assert  float(round(threshold,2)) == float(table.loc[totalRunID,"threshold"])
        else:
            ignoreUnMatch-=1

import matplotlib.pyplot as plt
data=pd.read_csv("/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/realtime/experiment/catalin_data_for_adaptive_thresold.csv")
data=data.dropna()
# data.iloc[-50:]
subjects = np.unique(list(data['sub']))
ignoreUnMatch=1 # found one mismatch, ignore this
performance=[]
for sub in subjects:
    print(sub)
    table = data[data["sub"]==sub]
    table=table.reset_index()
    plt.figure()
    t=list(table['progress'])
    performance.append(t)
    _=plt.plot(t)
    for totalRunID in range(len(table)):
        AdaptiveThreshold(totalRunID,table,ignoreUnMatch)
print("done")

# plot all traces
performances=np.zeros((10,50))
performances[performances==0]=None
for i,sub in enumerate(performance):
    plt.plot(sub)
    performances[i,0:len(sub)]=sub

plt.imshow(performances)

plt.plot(np.nanmean(performances,axis=0))
plt.title("progress trace mean for all subjects")
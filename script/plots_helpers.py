import matplotlib.pyplot as plt    
import numpy as np

def distributionsPlot(y,tX,featuresNames):
    savetX = tX
    savey = y
    alphaQuantile = 0

    for i in range(len(featuresNames)):

        y =  savey[(savetX[:,i] != - 999.0)]
        tX = savetX[(savetX[:,i] != - 999.0),:]
        
        if tX.shape[0]!=0:

            idPositive = [y==1][0]
            idNegative = [y==-1][0]

            plt.hist(tX[idPositive,i] ,100, histtype ='step',color='b',label='y == 1',density=True)      
            plt.hist(tX[idNegative,i] ,100, histtype ='step',color='r',label='y == -1',density=True)  
            plt.legend(loc = "upper right")
            plt.title("{name}, feature: {id}/{tot}".format(name=featuresNames[i],id=i,tot=len(featuresNames)-1), fontsize=12)
            plt.show()
            

def hist_plot_jet_class(y,tX):
    msk_jets_train = {
        0: tX[:, 22] == 0,
        1: tX[:, 22] == 1,
        2: tX[:, 22] == 2, 
        3: tX[:, 22] == 3
        }

    ax = plt.subplot(111)
    colors = ['b','g','r','y']
    legend = ['PRI_Jet_num: 0','PRI_Jet_num: 1','PRI_Jet_num: 2','PRI_Jet_num: 3']
    ind = np.array([-1,  1])
    w = 0.25
    for idx in range(len(msk_jets_train)):
        y_idx = y[msk_jets_train[idx]]
        count_prediction = {-1:  np.count_nonzero(y_idx == -1), 1:  np.count_nonzero(y_idx == 1)}
        ax.bar(ind+w*idx, count_prediction.values(), width=w, color=colors[idx],align='center')

    ax.set_ylabel('Numbers of training data')
    ax.set_xticks(ind+0.25)
    ax.set_xticklabels( ('y = -1', 'y = 1') )
    ax.legend(legend)
    ax.plot()
    
def hist_plot_best_methods():
    # best scores per jet
    
    jet_0 = [0.830, 0.830, 0.843, 0.843, 0.827, 0.829]
    jet_1 = [0.787, 0.785, 0.807, 0.807, 0.782, 0.785]
    jet_2 = [0.790, 0.789, 0.819, 0.819, 0.794, 0.782]
    jet_3 = [0.788, 0.787, 0.832, 0.832, 0.779, 0.770]
    jet_scores = [jet_0, jet_1, jet_2, jet_3]

    ax = plt.subplot(111)
    colors = ['b','g','r','y']
    legend = ['PRI_Jet_num: 0','PRI_Jet_num: 1','PRI_Jet_num: 2','PRI_Jet_num: 3']
    ind = np.array([-5, -3, -1,  1, 3, 5])
    w = 0.25
    for idx in range(len(jet_scores)):
        ax.bar(ind+w*idx, jet_scores[idx], width=w, color=colors[idx],align='center')

    ax.set_ylabel('Accuracy Score')
    ax.set_ylim([0.75, 0.88])
    ax.set_xticks(ind+w)
    ax.set_xticklabels( ('GD', 'SGD', 'LS', 'Ridge', 'Logistic', 'Reg. LR') )
    ax.legend(legend)
    ax.plot()
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
    legend = ['calss: 0','class: 1','class: 2','class: 3']
    ind = np.array([-1,  1])
    w = 0.25
    for idx in range(len(msk_jets_train)):
        y_idx = y[msk_jets_train[idx]]
        count_prediction = {-1:  np.count_nonzero(y_idx == -1), 1:  np.count_nonzero(y_idx == 1)}
        ax.bar(ind+w*idx, count_prediction.values(), width=w, color=colors[idx],align='center')

    ax.set_ylabel('Numbers of training data')
    ax.set_xticks(ind+0.25)
    ax.set_xticklabels( ('prediction is -1', 'prediction is 1') )
    ax.legend(legend)
    ax.plot()
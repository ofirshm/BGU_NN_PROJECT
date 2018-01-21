import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from sklearn.neural_network import MLPRegressor
from Cycle import Cycle
from galvani.galvani import *
from mpl_toolkits.mplot3d import Axes3D

def sample_to_cycle_vector(volt,current):
    mini = argrelextrema(volt, np.less)
    real_min = []
    cycle_vector = []
    last_data = 0
    real_min.append(last_data)
    for value in np.nditer(mini):
        if abs(value-last_data) > 2000:
            real_min.append(value)
            last_data = value
    for index in range(0,len(real_min)-1):
        cycle = Cycle(volt[real_min[index]:real_min[index+1]],current[real_min[index]:real_min[index+1]])
        cycle_vector.append(cycle)
    return cycle_vector


if __name__ == '__main__':
    cul1 = 'Ewe/V'
    cul2 = 'I/mA'
    x_train = []
    x_test = []
    # [CLOZAPINE , URIC ACID]
    y_train = [[0 ,0], [0.5 ,0 ], [2,0],[ 10,0], [0,0.15], [0.5,0.15],[2 , 0.15] , [10,0.15], [0,0.3], [0.5,0.3],[2 , 0.3] , [10,0.3], [0,0.5], [0.5,0.5],[2 , 0.5] , [10,0.5]]
    #y_train = [i/10 for i in y_train]
    y_true = [[0.1, 0.35], [3 , 0.3] ]

    feature_type=2 # NEXT CODE WILL BE ANALYSIS FOR THIS TYPE OF FEAURES :
    #Type 1 is 8 features for Oxidation peak
    #Type 2 is the feature for Signal avarage


    for solution_number in range(1,19):
        features_size = 90
        features = []
        for electrode in range(1,3):
            filename = 'Project Results/Sol#{0} ACV_C0{1}.mpr'.format(solution_number,electrode)
            mprfile = MPRfile(filename)
            current = mprfile.data[cul2]
            volt = mprfile.data[cul1]
            cycles = sample_to_cycle_vector(volt,current)
            # plt.clf()
            # plt.title("solution number: "+ repr(solution_number) + ", electrode number: " + repr(electrode))
            # cycles[0].plot_cycle()
            features.extend(cycles[0].get_features(feature_type,electrode))
            #features.extend(cycles[1].get_features(features_size))
            ###############
            # PLOT
            ###############



          #  plt.figure(1)
          #  plot_title = 'Sol#{0} Channel#{1}'.format(solution_number, electrode)
          #  plt.title(plot_title)
          #  cycles[0].plot_compared_current(10e6,0.06)

        if solution_number > 16 :
            x_test.append(features)
        else:
            x_train.append(features)

    interations=[10,100,200,400,700,1000]
    n_iter= len(interations)
    hidden_layers=[2,5,10,20,30,50,60,70]
    n_hidden=len(hidden_layers)
    i=1
    x=[]
    y=[]
    z=[]
    for interation in interations:
        for hidden_layer in hidden_layers:
                regressor = MLPRegressor(solver='sgd',alpha=1e-5,hidden_layer_sizes=(hidden_layer),random_state=1,max_iter=interation)
                regressor.fit(x_train,y_train)
                #y_predict=regressor.predict(x_test)

                x.append (hidden_layer)
                y.append(interation)
                z.append(regressor.score(x_test,y_true))



    xn= np.array(x)
    yn=np.array(y)
    zn=np.array(z)
    xnp = xn.reshape(n_iter,n_hidden)
    ynp = yn.reshape(n_iter,n_hidden)
    znp = zn.reshape(n_iter,n_hidden)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_wireframe(xnp, ynp, znp)
    ax.set_xlabel('Hidden Layer')
    ax.set_ylabel('Iterations')
    ax.set_zlabel('R square coefficient')
    plt.show()

    plt.show()
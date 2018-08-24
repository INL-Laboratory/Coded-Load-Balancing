# This file simulates the randomly selection of the server.
# The one choice and two choices are both implemented
# in this file that runs Simulator1 and Simulator2.
# Here we change the number of servers and find the average cost
# of users and maximum load of servers.
from __future__ import division
import math
#import matplotlib.pyplot as plt
#import networkx as nx
from multiprocessing import Pool
import sys
import numpy as np
import scipy.io as sio
import time
from CodedLoadBalancing.Simulator import *

#--------------------------------------------------------------------
#log = math.log
#sqrt = math.sqrt

#--------------------------------------------------------------------
# Simulation parameters:


# Choose the simulator. It can be the following values:
# 'OneChoice'
# 'TwoChoices'
simulator = 'Coded'
#simulator = 'TwoChoices'


# Base part of the output file name
base_out_filename = 'ChnkSzVar'


# Pool size for parallel processing
pool_size = 2


# Number of runs for computing average values. It is more eficcient that num_of_runs be a multiple of pool_size
num_of_runs = 10


# Number of servers
srv_num = 2025


# Cache size of each server (expressed in number of files)
cache_sz = 2


# Total number of files in the system
file_num = 100


# The number of chunks
chnk_num = 1
chnk_range = [1]

# The graph structure of the network
# It can be:
# 'Lattice' for square lattice graph. For the lattice the graph size should be perfect square.
# 'RGG' for random geometric graph. The RGG is generate over a unit square or unit cube.
# 'BarabasiAlbert' for Barabasi Albert random graph. It takes two parameters: # of nodes and # of edges
#       to attach from a new node to existing nodes
graph_type = 'Lattice'
#graph_type = 'RGG'
#graph_type = 'BarabasiAlbert'


# The parameters of the selected random graph
# The dictionary graph_param should always be defined. However, for some graphs it may not be used.
graph_param = {'rgg_radius': sqrt(5 / 4 * log(srv_num)) / sqrt(srv_num)} # if the graph is random geometric graph (RGG)
graph_param = {'num_edges' : 1} # For Barbasi Albert random graphs.


# The distribution of file placement in nodes' caches
# It can be:
# 'Uniform' for uniform placement.
# 'Zipf' for zipf distribution. We have to determine the parameter 'gamma' for this distribution in 'place_dist_param'.
placement_dist = 'Uniform'
#placement_dist = 'Zipf'


# The parameters of the placement distribution
place_dist_param = {'gamma' : 0.0}  # For Zipf distribution where 0 < gamma < infty


#--------------------------------------------------------------------
if __name__ == '__main__':
    # Create a number of workers for parallel processing
    pool = Pool(processes=pool_size)

    t_start = time.time()

    rslt_maxload = np.zeros((len(chnk_range), 1 + num_of_runs))
    rslt_avgcost = np.zeros((len(chnk_range), 1 + num_of_runs))
    rslt_outage = np.zeros((len(chnk_range), 1 + num_of_runs))
    for i, chnk_num in enumerate(chnk_range):
        req_num = srv_num
        params = [(srv_num, req_num, cache_sz, file_num, chnk_num, graph_type, graph_param, placement_dist, place_dist_param)
                  for itr in range(num_of_runs)]
        print(params)
        if simulator == 'Coded':
            rslts = pool.map(coded_load_balancing_simulator, params)
        elif simulator == 'TwoChoices':
            rslts = pool.map(simulator_twochoice, params)
        else:
            print('Error: an invalid simulator!')
            sys.exit()

        for j, rslt in enumerate(rslts):
            rslt_maxload[i, j + 1] = rslt['maxload']
            rslt_avgcost[i, j + 1] = rslt['avgcost']
            rslt_outage[i, j + 1] = rslt['outage']
            #tmpmxld = tmpmxld + rslt['maxload']
            #tmpavgcst = tmpavgcst + rslt['avgcost']

        rslt_maxload[i, 0] = chnk_num
        rslt_avgcost[i, 0] = chnk_num
        rslt_outage[i, 0] = chnk_num

    t_end = time.time()
    print("The runtime is {}".format(t_end-t_start))

    # Write the results to a matlab .mat file
    if placement_dist == 'Uniform':
        sio.savemat(base_out_filename + '_{}_{}_{}_sn={}_fn={}_cs={}_itr={}.mat'.\
            format(graph_type, placement_dist, simulator, srv_num, file_num, cache_sz, num_of_runs),
            {'maxload': rslt_maxload, 'avgcost': rslt_avgcost, 'outage': rslt_outage})
    elif placement_dist == 'Zipf':
        sio.savemat(base_out_filename + '_{}_{}_gamma={}_{}_sn={}_fn={}_cs={}_itr={}.mat'.\
            format(graph_type, placement_dist, place_dist_param['gamma'], simulator, srv_num, file_num, cache_sz, num_of_runs),
            {'maxload': rslt_maxload, 'avgcost': rslt_avgcost, 'outage': rslt_outage})



#    if simulator == 'one choice':
#        sio.savemat(base_out_filename + '_{}_fn={}_cs={}_itr={}.mat'.format(simulator, file_num, cache_sz, num_of_runs),
#                    {'maxload': rslt_maxload, 'avgcost': rslt_avgcost, 'outage':rslt_outage})
#    elif simulator == 'two choice':
#        sio.savemat(base_out_filename + '_two_choice_' + 'fn={}_cs={}_itr={}.mat'.format(file_num, cache_sz, num_of_runs),
#                    {'maxload': rslt_maxload, 'avgcost': rslt_avgcost, 'outage':rslt_outage})





#    if simulator == 'one choice':
#        np.savetxt(base_out_filename+'_sim1_'+'fn={}_cs={}_itr={}.txt'.format(file_num,cache_sz,num_of_runs), result, delimiter=',')
#    elif simulator == 'two choice':
#        np.savetxt(base_out_filename+'_sim2_'+'fn={}_cs={}_itr={}.txt'.format(file_num,cache_sz,num_of_runs), result, delimiter=',')

#    plt.plot(result[:,0], result[:,1])
#    plt.plot(result[:,0], result[:,2])
#    plt.xlabel('Cache size in each server (# of files)')
#    plt.title('Random server selection methods. Number of servers = {}, number of files = {}'.format(srv_num,file_num))
#    plt.show()

    #print(result['loads'])
    #print("------")
    #print('The maximum load is {}'.format(result['maxload']))
    #print('The average request cost is {0:0}'.format(result['avgcost']))

    print('Outage:')
    print(rslt_outage)

    print('Maxload:')
    print(rslt_maxload)

    print('Communication Cost:')
    print(rslt_avgcost)
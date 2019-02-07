# This is the main file to run the simulator for our coded load balancing scheme.
#
# Here we change the number of chunks a file is partition to and simulate the average cost
# of users and the maximum load of servers.
#
#
from __future__ import division
import math
from multiprocessing import Pool
import sys
import numpy as np
import scipy.io as sio
import time
from CodedLoadBalancing.Simulator_MltChnk import *


#--------------------------------------------------------------------
# Simulation parameters:


# Choose the simulator. It can be the following values:
simulator = 'Coded'


# Base part of the output file name
base_out_filename = 'ChnkSzVar'


# Pool size for parallel processing
pool_size = 1


# Number of runs for computing average values. It is more eficcient that num_of_runs be a multiple of pool_size
num_of_runs = 1


# Number of servers
srv_num = 49


# Cache size of each server (expressed in number of files)
cache_sz = 1


# Total number of files in the system
file_num = 10


# The number of chunks
#     If the number of chunks is set to one, we in fact simulate the nearest replica strategy.
# Here, instead of determining the 'number of chunks,' we provide a set of chunk numbers in the
#     variable 'chnk_range'.
#chnk_num = 5
#chnk_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
chnk_range = [5]


# The maximum number of chunks that can be downloaded from each server.
chnk_max = 10


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
        params = [(srv_num, req_num, cache_sz, file_num, chnk_num, chnk_max, graph_type, graph_param, placement_dist, place_dist_param)
                  for itr in range(num_of_runs)]
        print(params)
        if simulator == 'Coded':
            rslts = pool.map(coded_load_balancing_simulator, params)
#        elif simulator == 'TwoChoices':
#            rslts = pool.map(simulator_twochoice, params)
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
        sio.savemat(base_out_filename + '_{}_{}_{}_sn={}_fn={}_cs={}_chnmax={}_itr={}.mat'.\
            format(graph_type, placement_dist, simulator, srv_num, file_num, cache_sz, chnk_max, num_of_runs),
            {'maxload': rslt_maxload, 'avgcost': rslt_avgcost, 'outage': rslt_outage})
    elif placement_dist == 'Zipf':
        sio.savemat(base_out_filename + '_{}_{}_gamma={}_{}_sn={}_fn={}_cs={}_chnmax={}_itr={}.mat'.\
            format(graph_type, placement_dist, place_dist_param['gamma'], simulator, srv_num,\
                   file_num, cache_sz, chnk_max, num_of_runs),\
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
# This file contains the coded load balancing simulators for distributed load balancing
#   in a content delivery network (CDN):
#
#
# [1]: Add a ref to our paper
#
from __future__ import division
import math
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from CodedLoadBalancing.Statistic import bounded_zipf
from CodedLoadBalancing.Server import Server
from CodedLoadBalancing.Graph import *


#--------------------------------------------------------------------
log = math.log
sqrt = math.sqrt


#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
# This function simulates coded scheme proposed in our paper.
# Here each request is served by receiving 'chnk_num (l in the paper)' number of coded chunks from the nearest servers
#   cached the coded chunks of the requested file.
# When chnk_num = 1, the simulator will reduce to the 'one choice (Nearest Replica)' strategy we had in our previous
#   paper.
#
def coded_load_balancing_simulator(params):
    # Simulation parameters
    # srv_num: Number of servers
    # req_num: Number of requests
    # cache_sz: Cache size of each server (expressed in number of files)
    # chnk_num: Number of chunks each file is partitioned
    # chnk_max: Maximum number of chunks that can be downloaded from each server.
    # file_num: Total number of files in the network
    # graph_type: topology of the network
    # graph_param: graph parameter of a specific network topology (in fact parameters of a random ensemble the topology
    #               is taken from
    # placement_dist: random distribution that should be used for file placement in the servers' caches
    # place_dist_param: parameter of a specific random distribution (e.g., gamma for Zipf distribution)

    np.random.seed()  # to prevent all the instances produce the same results!

    srv_num, req_num, cache_sz, file_num, chnk_num, chnk_max, graph_type, graph_param, placement_dist, place_dist_param = params

    print('The "coded simulator" is starting with parameters:')
    print('# of Servers = {}, # of Reqs = {}, # of Files = {}, Cache Size = {}, # of Chunks = {}, Max Chunks = {}\
            Net. Topology = {}, Placement Dist. = {}, Plc. Dist. Param. = {}'\
            .format(srv_num, req_num, file_num, cache_sz, chnk_num, chnk_max, graph_type, placement_dist, place_dist_param))

    # Check the validity of parameters
#    if cache_sz > file_num:
#        print("Error: The cache size is larger that number of files!")
#        sys.exit()

    # Check if the total cache memory in the system is larger than the file library size.
    if srv_num * cache_sz < file_num:
        print("Error: The total cache memory of the system is less than the size of file library!")
        sys.exit()

    if graph_type == 'Lattice':
        side = sqrt(srv_num)
        if not side.is_integer():
            print("Error: the size of lattice is not perfect square!")
            sys.exit()
        shortest_path_matrix = all_shortest_path_length_torus(srv_num)
#        print('Start generating a square lattice graph with {} nodes...'.format(srv_num))
#        G = Gen2DLattice(srv_num)
#        print('Succesfully generates a square lattice graph with {} nodes...'.format(srv_num))
    elif graph_type == 'RGG': # if the graph is random geometric graph (RGG)
        rgg_radius = graph_param['rgg_radius']
        # Generate a random geometric graph.
        print('Start generating a random geometric graph with {} nodes...'.format(srv_num))
        conctd = False
        while not conctd:
            G = nx.random_geometric_graph(srv_num, rgg_radius)
            conctd = nx.is_connected(G)
        all_sh_len = nx.all_pairs_shortest_path_length(G)
        shortest_path_matrix = np.array([all_sh_len[i][j] for i in all_sh_len.keys() for j in all_sh_len[i].keys()],\
                                        dtype=np.int32)
        shortest_path_matrix = shortest_path_matrix.reshape(srv_num, srv_num)
        print('Succesfully generates a connected random geometric graph with {} nodes...'.format(srv_num))
    elif graph_type == 'BarabasiAlbert': # if the graph is generated according to Barabasi and Albert model.
        num_edges = graph_param['num_edges']
        print('Start generating a Barabasi-Albert graph with {} nodes and edge parameter {} ...'\
              .format(srv_num, num_edges))
        G = nx.barabasi_albert_graph(srv_num, num_edges)
        all_sh_len = nx.all_pairs_shortest_path_length(G)
        shortest_path_matrix = np.array([all_sh_len[i][j] for i in all_sh_len.keys() for j in all_sh_len[i].keys()],\
                                        dtype=np.int32)
        shortest_path_matrix = shortest_path_matrix.reshape(srv_num, srv_num)
        print('Succesfully generates a Barabasi-Albert graph with {} nodes and edge parameter {} ...'\
              .format(srv_num, num_edges))
    else:
        print("Error: the graph type is not known!")
        sys.exit()
    # Draw the graph
    #print(shortest_path_matrix)
    #nx.draw(G)
    #plt.show()
    #sys.exit()

    # Create 'srv_num' (empty or null) servers from the class server
    srvs = [Server(i) for i in range(srv_num)]

    # Cache placement at servers according to file popularity distribution
    #
    # file_sets = List of sets of servers containing at least a chunk from each file
    # list_cached_files = List of all cached files at the end of placement process
    srvs, file_sets, list_cached_files = \
        srv_cache_placement(srv_num, file_num, cache_sz, chnk_num, placement_dist, place_dist_param, srvs)
    #print(list_cached_files)

    # Truncate "file_sets" to only allow download "chnk_max" chunks from each server.
    truncated_file_sets = [[] for i in range(file_num)]
    for i in np.arange(file_num, dtype=np.int32):
        srv_cnt_tmp = np.zeros(srv_num)
        for j in np.arange(len(file_sets[i]), dtype=np.int32):
            srv_cnt_tmp[file_sets[i][j]] += 1
            if srv_cnt_tmp[file_sets[i][j]] <= chnk_max:
                truncated_file_sets[i].append(file_sets[i][j])

    # ---------------------------------------------------------------
    # Main loop of the simulator.
    # ---------------------------------------------------------------
    # We throw 'req_num' of balls (requests) into the servers.
    # Each request randomly picks a server and a file.

    print('The "coded simulator" core is starting...')

    total_cost = 0  # Total communication cost, measured in number of hops.
    outage_num = 0  # Measure the # of outage events, i.e., the # of events that the requested
                    # file has not completely cached in the whole network.

    # Generate all the random incoming servers (i.e., incoming events are received at these servers)
    incoming_srv_lst = np.random.randint(srv_num, size=req_num)

    # Generate all the random requests
    if placement_dist == 'Uniform':
        rqstd_file_lst = np.random.randint(file_num, size=req_num)
    elif placement_dist == 'Zipf':
        rqstd_file_lst = bounded_zipf(file_num, place_dist_param['gamma'], req_num)
#        print(rqstd_file_lst)

    for i in np.arange(req_num, dtype=np.int32):
        #print(i)
        incoming_srv = incoming_srv_lst[i] # i'th Random incoming server

        rqstd_file = rqstd_file_lst[i] # i'th Random requested file
        #print('i=',i,"incm srv=",incoming_srv,"rqstd file=",rqstd_file)

#        print(file_sets)
        if (rqstd_file in list_cached_files) and (len(file_sets[rqstd_file]) >= chnk_num): # not sure about this condition
            #print(shortest_path_matrix[incoming_srv, :][file_sets[rqstd_file]])
            #print(np.argpartition(shortest_path_matrix[incoming_srv, :][file_sets[rqstd_file]], chnk_num-1))
            # Find the 'chnk_num' nearest servers that have a chunk from the requested file 'rqstd_file'

#            print('shortest path = {}'.format(shortest_path_matrix[incoming_srv, :][file_sets[rqstd_file]]))
            indx_nearest_srvs = \
                np.argsort(shortest_path_matrix[incoming_srv, :][file_sets[rqstd_file]])[0:chnk_num] # either of these line should work
                #np.argpartition(shortest_path_matrix[incoming_srv, :][file_sets[rqstd_file]], chnk_num-1)[0:chnk_num]
            tmp1 = shortest_path_matrix[incoming_srv, :]
            tmp2 = shortest_path_matrix[incoming_srv, :][file_sets[rqstd_file]]

            nearest_srvs = np.array(file_sets[rqstd_file])[indx_nearest_srvs]
#            print(file_sets[rqstd_file])
#            print(shortest_path_matrix[incoming_srv, :])
#            print(shortest_path_matrix[incoming_srv, :][file_sets[rqstd_file]])
#            print(indx_nearest_srvs)
#            print("nearest srv: ", nearest_srvs)
#            print('\n')
            for s in nearest_srvs:
                srvs[s].add_load(1.0/chnk_num)
                total_cost += (1.0/chnk_num) * shortest_path_matrix[incoming_srv, s]
        else:
            outage_num += 1


            # Implement random placement without considering loads
##            srvs[srv0].add_load()
##        else:
##            outage_num += 1

    print('The "coded simulator" is done.')

    # At the end of simulation, find maximum load, etc.
    loads = [srvs[i].get_load() for i in np.arange(srv_num, dtype=np.int32)]
    maxload = max(loads)
    avgcost = total_cost / req_num
    outage = outage_num / req_num

#    print(loads)

    return {'loads':loads, 'maxload':maxload, 'avgcost':avgcost, 'outage':outage}






#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
# This function randomly places library files into the server caches.
#
# Input:
#     srv_num = number of servers
#     file_num = number of files in the library
#     cache_sz = size of cache in each server
#     chnk_num = number of chunks each file is divided to
#     placement_dist = the file popularity distribution
#     srvs = A list of 'srv_num' (empty or null) servers from the class server
#
# Return:
#     srvs = A list of 'srv_num' servers from the class server
#     file_sets = List of sets of servers containing at least a chunk from each file
#     list_cached_files = List of all cached files at the end of placement process
#
def srv_cache_placement(srv_num, file_num, cache_sz, chnk_num, placement_dist, place_dist_param, srvs):

    # List of sets of servers containing a chunk from each file. The list is enumerate by the
    #     file index and for each file we have a list that contains the servers cached a chunk from that file.
    file_sets = [[] for i in range(file_num)]

    # List of all cached files at the end of placement process
    list_cached_files = []

    # The index of chunks created (used) so far for each file. The list is indexed by the file indices.
    chnk_used_so_far = [-1 for i in range(file_num)]

    if placement_dist == 'Uniform':
        # Randomly places cache_sz files in each servers
##        print('Start randomly placing {} files in each server with Uniform dist.'.format(cache_sz))

        # First put randomly 'chnk_num' chunks of each file in a subset of servers
        #     (assuming file_num =< srv_num * cache size).
        #     This means that at least one replication from each file is cached in the network.
        all_chunks = [ [fl, chnk] for fl in range(file_num) for chnk in range(chnk_num) ]
#        print(all_chunks)
        all_chunks = np.random.permutation(all_chunks).tolist()
#        print('all_chunks = {}'.format(all_chunks))
        #lst = np.random.permutation(srv_num)[0:file_num]
        for i in np.arange(file_num * chnk_num, dtype=np.int32):
            chnk_used_so_far[all_chunks[i][0]] += 1
            srv_indx = i % srv_num
            file_sets[all_chunks[i][0]].append(srv_indx)
#(2)            if srv_indx not in file_sets[all_chunks[i][0]]:
#(2)                file_sets[all_chunks[i][0]].append(srv_indx)
#(1)         file_sets[all_chunks[i][0]].append(srv_indx)
            srvs[srv_indx].append_files_list(all_chunks[i])
#            print(all_chunks[i])

       # print('chnk_used_so_far = {}'.format(chnk_used_so_far))
       # print('file_sets = {}'.format(file_sets))
        #for i, s in enumerate(lst):
        #    file_sets[i].append(s)
        #    srvs[s].set_files_list([i])
        #    list_cached_files.append(i)

        # Then fills the rest of empty cache places
        for srv_indx in np.arange(srv_num, dtype=np.int32):
            #print(s)
            srv_fl_lst = srvs[srv_indx].get_files_list()
#            print(srvlst)
            #tmp_lst = np.random.permutation(file_num)[0:cache_sz - len(srvlst)]
            tmp_lst = np.random.randint(0, file_num, cache_sz * chnk_num - len(srv_fl_lst))
            #srvs[s].set_files_list(srvlst.extend(tmp_lst))
#            for j in range(len(tmp_lst)):
            for fl in tmp_lst:
                chnk_used_so_far[fl] += 1
                srvs[srv_indx].append_files_list([ fl, chnk_used_so_far[fl] ])
#                if srv_indx not in file_sets[fl]:
#                    file_sets[fl].append(srv_indx)
                file_sets[fl].append(srv_indx)

        # Fill the list of cached files
        #for i in range(file_num):
        #    if chnk_used_so_far >= chnk_num - 1:
        #        list_cached_files.append(i)

###        print('Done with randomly placing {} files in each server with Uniform dist.'.format(cache_sz))
#        print(chnk_used_so_far)
        #for s in range(srv_num):
        #    print(srvs[s].get_files_list())
    elif placement_dist == 'Zipf':
        # Randomly places 'cache_sz * chnk_num' chunks in each servers with Zipf dist.
        gamma = place_dist_param['gamma']  # Parameter of Zipf distribution.
###        print('Start randomly placing {} chunks in each server according to Zipf dist. with gamma={}'.\
###              format(cache_sz*chnk_num, gamma))

        for s in np.arange(srv_num, dtype=np.int32):
            # Generates some sample from a zipf distribution over the set {1,...,file_num}
            tmp_zipf_smpl = bounded_zipf(file_num, gamma, cache_sz*chnk_num)
            for j in np.arange(len(tmp_zipf_smpl), dtype=np.int32):
                chnk_used_so_far[tmp_zipf_smpl[j]] += 1
                srvs[s].append_files_list([ tmp_zipf_smpl[j], chnk_used_so_far[tmp_zipf_smpl[j]] ])
#                if s not in file_sets[tmp_zipf_smpl[j]]:
#                    file_sets[tmp_zipf_smpl[j]].append(s)
                file_sets[tmp_zipf_smpl[j]].append(s)

        #    srvs[s].set_files_list(tmp_lst)
        #    for j in np.arange(len(tmp_lst), dtype=np.int32):
        #        if s not in file_sets[tmp_lst[j]]:
        #            file_sets[tmp_lst[j]].append(s)
        #        if tmp_lst[j] not in list_cached_files:
        #            list_cached_files.append(tmp_lst[j])
        #print(list_chached_files)
###        print('Done with randomly placing {} files in each server with Zipf dist.'.format(cache_sz))

    # Fill the list of cached files
    for i in range(file_num):
        if chnk_used_so_far[i] >= chnk_num - 1:
            list_cached_files.append(i)

    # Sort the entries of file_sets for each file
        #for i in np.arange(file_num, dtype=np.int32):
        #    file_sets[i] = np.sort(file_sets[i])

#    for s in range(srv_num):
#        print(srvs[s].get_files_list())

    #print('list_cached_files = {}'.format(list_cached_files))
#    print(len(list_cached_files))

#    print(chnk_used_so_far)
#    print(len(chnk_used_so_far))
#    print(sum(chnk_used_so_far))


    return srvs, file_sets, list_cached_files
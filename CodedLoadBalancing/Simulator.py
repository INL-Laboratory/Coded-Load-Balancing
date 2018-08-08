# This file containes 3 different simulators for distributed load balancing in a content delivery network (CDN):
# These 3 different simulators are:
#   clb_simulator_onechoice: which implements the naive "nearest replica strategy" explained in [1],
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
#import string
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
    # file_num: Total number of files in the network
    # graph_type: topology of the network
    # graph_param: graph parameter of a specific network topology (in fact parameters of a random ensemble the topology
    #               is taken from
    # placement_dist: random distribution that should be used for file placement in the servers' caches
    # place_dist_param: parameter of a specific random distribution (e.g., gamma for Zipf distribution)

    np.random.seed()  # to prevent all the instances produce the same results!

    srv_num, req_num, cache_sz, file_num, chnk_num, graph_type, graph_param, placement_dist, place_dist_param = params

    print('The "coded simulator" is starting with parameters:')
    print('# of Servers = {}, # of Reqs = {}, # of Files = {}, Cache Size = {}, # of Chunks = {},\
            Net. Topology = {}, Placement Dist. = {}, Plc. Dist. Param. = {}'\
            .format(srv_num, req_num, file_num, cache_sz, chnk_num, graph_type, placement_dist, place_dist_param))

    # Check the validity of parameters
    if cache_sz > file_num:
        print("Error: The cache size is larger that number of files!")
        sys.exit()

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

    # Create 'srv_num' servers from the class server
    srvs = [Server(i) for i in range(srv_num)]

    # Cache placement at servers according to file popularity distribution
    #
    # file_sets = List of sets of servers containing each file
    # list_cached_files = List of all cached files at the end of placement process
    srvs, file_sets, list_cached_files = \
        srv_cache_placement(srv_num, file_num, cache_sz, chnk_num, placement_dist, place_dist_param, srvs)
    #print(list_cached_files)

    # ---------------------------------------------------------------
    # Main loop of the simulator.
    # ---------------------------------------------------------------
    # We throw n balls requests into the servers.
    # Each request randomly pick a server and a file.
    print('The "coded simulator" core is starting...')

    total_cost = 0  # Measured in number of hops .
    outage_num = 0  # Measure the # of outage events, i.e., the # of events that the requested
                    # file has not cached in the whole network.

    # Generate all the random incoming servers (i.e., incoming events are received at these servers)
    incoming_srv_lst = np.random.randint(srv_num, size=req_num)

    # Generate all the random requests
    if placement_dist == 'Uniform':
        rqstd_file_lst = np.random.randint(file_num, size=req_num)
    elif placement_dist == 'Zipf':
        rqstd_file_lst = bounded_zipf(file_num, place_dist_param['gamma'], req_num)
        print(rqstd_file_lst)

    for i in range(req_num):
        #print(i)
        incoming_srv = incoming_srv_lst[i] # i'th Random incoming server

        rqstd_file = rqstd_file_lst[i] # i'th Random requested file
        #print('i=',i,"incm srv=",incoming_srv,"rqstd file=",rqstd_file)

        if (rqstd_file in list_cached_files) and (len(file_sets[rqstd_file]) >= chnk_num):
            #print(shortest_path_matrix[incoming_srv, :][file_sets[rqstd_file]])
            #print(np.argpartition(shortest_path_matrix[incoming_srv, :][file_sets[rqstd_file]], chnk_num-1))
            # Find the 'chnk_num' nearest servers that have a chunk from the requested file 'rqstd_file'
            indx_nearest_srvs = \
                np.argpartition(shortest_path_matrix[incoming_srv, :][file_sets[rqstd_file]], chnk_num-1)[0:chnk_num]
            #print(indx_nearest_srvs)
            nearest_srvs = np.array(file_sets[rqstd_file])[indx_nearest_srvs]
            print("nearest srv: ", nearest_srvs)
            for s in nearest_srvs:
                srvs[s].add_load(1.0/chnk_num)
                total_cost += (1.0 / chnk_num) * shortest_path_matrix[incoming_srv, s]
        else:
            outage_num += 1

##        if rqstd_file in list_cached_files:
##            if incoming_srv in file_sets[rqstd_file]:
##                srv0 = incoming_srv
##            else:
                # Find the nearest server that has the requested file
                #all_sh_path_len_G = nx.shortest_path_length(G, source=incoming_srv)
                #all_sh_path_len_G = shortest_path_length_torus(srv_num, incoming_srv)
##                dmin = 2*srv_num # some large number larger that the maximum diameter of the topology!
##                for nd in file_sets[rqstd_file]:
                    #d = nx.shortest_path_length(G, source=incoming_srv, target=nd)
                    #d = all_sh_path_len_G[nd]
##                    d = shortest_path_matrix[incoming_srv, nd]
##                    if d < dmin:
##                        dmin = d
##                        srv0 = nd
##                total_cost = total_cost + dmin
#                srv0 = file_sets[rqstd_file][np.random.randint(len(file_sets[rqstd_file]))]
#                total_cost = total_cost + nx.shortest_path_length(G, source=incoming_srv, target=srv0)

            # Implement random placement without considering loads
##            srvs[srv0].add_load()
##        else:
##            outage_num += 1

    print('The "coded simulator" is done.')

    # At the end of simulation, find maximum load, etc.
    loads = [srvs[i].get_load() for i in range(srv_num)]
    maxload = max(loads)
    avgcost = total_cost / srv_num
    outage = outage_num / srv_num

#    print(loads)

    return {'loads':loads, 'maxload':maxload, 'avgcost':avgcost, 'outage':outage}


#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
# This function is the core implementation of power of two choices.
def simulator_twochoice(params):
    # Simulation parameters
    # srv_num: Number of servers
    # cache_sz: Cache size of each server (expressed in number of files)
    # file_num: Total number of files in the system

    np.random.seed()  # to prevent all the instances produce the same results!

    #srv_num, req_num, cache_sz, file_num, graph_type, placement_dist, place_dist_param = params
    srv_num, req_num, cache_sz, file_num, graph_type, graph_param, placement_dist, place_dist_param = params

    print('The "two choices" simulator is starting with parameters:')
    print('# of Servers = {}, # of Reqs = {}, # of Files = {}, Cache Size = {},\
            Net. Topology = {}, Placement Dist. = {}, Plc. Dist. Param. = {}'\
            .format(srv_num, req_num, file_num, cache_sz, graph_type, placement_dist, place_dist_param))

    # Check the validity of parameters
    if cache_sz > file_num:
        print("Error: The cache size is larger that number of files!")
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
    elif graph_type == 'RGG':  # if the graph is random geometric graph (RGG)
        rgg_radius = graph_param['rgg_radius']
        # Generate a random geometric graph.
        print('Start generating a random geometric graph with {} nodes...'.format(srv_num))
        conctd = False
        while not conctd:
            G = nx.random_geometric_graph(srv_num, rgg_radius)
            conctd = nx.is_connected(G)
        all_sh_len = nx.all_pairs_shortest_path_length(G)
        shortest_path_matrix = np.array([all_sh_len[i][j] for i in all_sh_len.keys() for j in all_sh_len[i].keys()], \
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
        # nx.draw(G)
        # plt.show()

    # Create 'srv_num' servers from the class server
    srvs = [Server(i) for i in range(srv_num)]

    # Cache placement at servers according to file popularity distribution
    #
    # file_sets = List of sets of servers containing each file
    # list_cached_files = List of all cached files at the end of placement process
    srvs, file_sets, list_cached_files = \
        srv_cache_placement(srv_num, file_num, cache_sz, placement_dist, place_dist_param, srvs)

    # ---------------------------------------------------------------
    # Main loop of the simulator.
    # ---------------------------------------------------------------
    # We throw n ball requests into the servers.
    # Each request randomly pick a server and a file.
    print('The "two choices" simulator core is starting...')

    total_cost = 0  # Measured in number of hops.
    outage_num = 0  # Measure the # of outage events, i.e., the # of events that the requested
                    # file has not cached in the whole network.

    # Generate all the random incoming servers
    incoming_srv_lst = np.random.randint(srv_num, size=req_num)

    # Generate all the random requests
    if placement_dist == 'Uniform':
        rqstd_file_lst = np.random.randint(file_num, size=req_num)
    elif placement_dist == 'Zipf':
        rqstd_file_lst = bounded_zipf(file_num, place_dist_param['gamma'], req_num)

#    print(len(rqstd_file_lst))
#    print(len(incoming_srv_lst))

    for i in range(req_num):
        #print(i)
        incoming_srv = incoming_srv_lst[i] # Random incoming server

        rqstd_file = rqstd_file_lst[i] # Random requested file

        #print(len(list_cached_files))

        if rqstd_file in list_cached_files:
#            all_sh_path_len_G = nx.shortest_path_length(G, source=incoming_srv)
            #all_sh_path_len_G = shortest_path_length_torus(srv_num, incoming_srv)
            if incoming_srv in file_sets[rqstd_file]:
                srv0 = incoming_srv
            else:
                # Find the nearest server that has the requested file
                dmin = 2*srv_num # some large number!
                for nd in file_sets[rqstd_file]:
                    #d = nx.shortest_path_length(G, source=incoming_srv, target=nd)
                    #d = all_sh_path_len_G[nd]
                    d = shortest_path_matrix[incoming_srv, nd]
                    if d < dmin:
                        dmin = d
                        srv0 = nd
#                srv0 = file_sets[rqstd_file][np.random.randint(len(file_sets[rqstd_file])+1) - 1]

            srv1 = srv0
            if len(file_sets[rqstd_file]) > 1:
                while srv1 == srv0:
                    srv1 = file_sets[rqstd_file][np.random.randint(len(file_sets[rqstd_file]))]

            # Implement power of two choices
            load0 = srvs[srv0].get_load()
            load1 = srvs[srv1].get_load()
            if load0 > load1:
                srvs[srv1].add_load()
                #total_cost += nx.shortest_path_length(G, source=incoming_srv, target=srv1)
                #total_cost += all_sh_path_len_G[srv1]
                total_cost += shortest_path_matrix[incoming_srv, srv1]
            elif load0 < load1:
                srvs[srv0].add_load()
                if srv0 != incoming_srv:
                    #total_cost += nx.shortest_path_length(G, source=incoming_srv, target=srv0)
                    #total_cost += all_sh_path_len_G[srv0]
                    total_cost += shortest_path_matrix[incoming_srv, srv0]
            elif np.random.randint(2) == 0:
                srvs[srv0].add_load()
                if srv0 != incoming_srv:
                    #total_cost += nx.shortest_path_length(G, source=incoming_srv, target=srv0)
                    #total_cost += all_sh_path_len_G[srv0]
                    total_cost += shortest_path_matrix[incoming_srv, srv0]
            else:
                srvs[srv1].add_load()
                #total_cost += nx.shortest_path_length(G, source=incoming_srv, target=srv1)
                #total_cost += all_sh_path_len_G[srv1]
                total_cost += shortest_path_matrix[incoming_srv, srv1]
        else:
            outage_num +=  1


    print('The "two choices" simulator is done.')
#    print('------------------------------')

    # At the end of simulation, find maximum load, etc.
    loads = [srvs[i].get_load() for i in range(srv_num)]
    maxload = max(loads)
    avgcost = total_cost/srv_num
    outage = outage_num / srv_num

#    print(loads)
#    print(maxload)
#    print(total_cost)

    return {'loads':loads, 'maxload':maxload, 'avgcost':avgcost, 'outage':outage}


#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
# This function simulates the trade-off between 'minimum
# communication cost scheme' and 'minimum load scheme'.
#
# Here we implement the power of two choices. However we change the
# radius of search for finding two bins. By changing this radius, we
# will get different trade-offs.
def simulator_tradeoff(params):
    # Simulation parameters
    # srv_num: Number of servers
    # cache_sz: Cache size of each server (expressed in number of files)
    # file_num: Total number of files in the system
    # graph_type: Topology of network
    # graph_param: Parameters for generating network topology
    # placement_dist: File library distribution
    # place_dist_param: Parameters of file library distribution
    # alpha: alpha determines the search space as follows: (1+alpha)NearestNeighborDistance

    np.random.seed()  # to prevent all the instances produce the same results!

    srv_num, cache_sz, file_num, graph_type, graph_param, placement_dist, place_dist_param, alpha = params

    print('The "Trade-off" simulator is starting with parameters:')
    print('# of Srvs = {}, # of Files = {}, Cache Size = {}, Net. Top. = {},\
            alpha = {}, Plc. Dist. = {}, Plc. Dist. Param. = {}'\
            .format(srv_num, file_num, cache_sz, graph_type, alpha, placement_dist, place_dist_param))

    # Check the validity of parameters
    if cache_sz > file_num:
        print("Error: The cache size is larger that number of files!")
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
    elif graph_type == 'RGG':  # if the graph is random geometric graph (RGG)
        rgg_radius = graph_param['rgg_radius']
        # Generate a random geometric graph.
        print('Start generating a random geometric graph with {} nodes...'.format(srv_num))
        conctd = False
        while not conctd:
            G = nx.random_geometric_graph(srv_num, rgg_radius)
            conctd = nx.is_connected(G)
        all_sh_len = nx.all_pairs_shortest_path_length(G)
        shortest_path_matrix = np.array([all_sh_len[i][j] for i in all_sh_len.keys() for j in all_sh_len[i].keys()], \
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
        # nx.draw(G)
        # plt.show()

    # Create 'srv_num' servers from the class server
    srvs = [Server(i) for i in range(srv_num)]

    # Cache placement at servers according to file popularity distribution
    #
    # file_sets = List of sets of servers containing each file
    # list_cached_files = List of all cached files at the end of placement process
    srvs, file_sets, list_cached_files = \
        srv_cache_placement(srv_num, file_num, cache_sz, placement_dist, place_dist_param, srvs)


    # List of sets of servers containing each file. The list is enumerate by the
    # file index and for each file we have a list that contains the servers cached that file.
#    file_sets = [[] for i in range(file_num)]

#    if placement_dist == 'Uniform':
        # Randomly places cache_sz files in each servers
#        print('Start randomly placing {} files in each server with Uniform dist.'.format(cache_sz))
        # First put randomly one copy of each file in a subset of servers (assuming file_num =< srv_num)
#        lst = np.random.permutation(srv_num)[0:file_num]
#        for i, s in enumerate(lst):
#            file_sets[i].append(s)
#            srvs[s].set_files_list([i])
        # Then fills the rest of empty cache places
#        for s in range(srv_num):
            #print(s)
#            srvlst = srvs[s].get_files_list()
#            tmp_lst = np.random.permutation(file_num)[0:cache_sz - len(srvlst)]
#            srvs[s].set_files_list(srvlst.extend(tmp_lst))
#            for j in range(len(tmp_lst)):
#                if s not in file_sets[tmp_lst[j]]:
#                    file_sets[tmp_lst[j]].append(s)
#        print('Done with randomly placing {} files in each server with Uniform dist.'.format(cache_sz))
#    elif placement_dist == 'Zipf':
        # Randomly places cache_sz files in each servers with Zipf dist.
#        print('Start randomly placing {} files in each server with Zipf dist.'.format(cache_sz))
#        gamma = place_dist_param['gamma']  # Parameter of Zipf distribution.
#        list_chached_files = []  # List of all cached files at the end of placement process

#        for s in np.arange(srv_num, dtype=np.int32):
#            tmp_lst = bounded_zipf(file_num, gamma, cache_sz)
#            srvs[s].set_files_list(tmp_lst)
#            for j in np.arange(len(tmp_lst), dtype=np.int32):
#                if s not in file_sets[tmp_lst[j]]:
#                    file_sets[tmp_lst[j]].append(s)
#                if tmp_lst[j] not in list_chached_files:
#                    list_chached_files.append(tmp_lst[j])
        #print(list_chached_files)
#        print('Done with randomly placing {} files in each server with Zipf dist.'.format(cache_sz))

    # Main loop of the simulator.
    # We throw n balls requests into the servers.
    # Each request randomly pick a server and a file.
    # Then we apply the power of two choices scheme but up to some radius from the requsting
    # node.
    print('The "Trade-off" simulator core is starting...')
#          with parameters: sn={}, cs={}, fn={}, graph_type={}, alpha={}'\
#          .format(srv_num, cache_sz, file_num, graph_type, alpha))

    total_cost = 0  # Measured in number of hops.
    outage_num = 0  # Measure the # of outage events, i.e., the # of events that the requested
                    # file has not cached in the whole network.

    # Generate all the random incoming servers
    incoming_srv_lst = np.random.randint(srv_num, size=srv_num)

    # Generate all the random requests
    if placement_dist == 'Uniform':
        rqstd_file_lst = np.random.randint(file_num, size=srv_num)
    elif placement_dist == 'Zipf':
        rqstd_file_lst = bounded_zipf(file_num, place_dist_param['gamma'], srv_num)

    for i in range(srv_num):
        #print(i)
        incoming_srv = incoming_srv_lst[i] # Random incoming server

        rqstd_file = rqstd_file_lst[i] # Random requested file

        if rqstd_file in list_cached_files:
            # Find the shortest path from requesting nodes to all other nodes
            #all_sh_path_len_G = nx.shortest_path_length(G, source=incoming_srv)
            #all_sh_path_len_G = shortest_path_length_torus(srv_num, incoming_srv)

            # Find the nearest server that has the requested file
#            print(file_sets[rqstd_file])
#            print(type(file_sets[rqstd_file][0]))
#            print(list(file_sets[rqstd_file]))
            nearest_neighbor = []
            srv_lst_for_this_request_excld_incom_srv = list(file_sets[rqstd_file])
            if incoming_srv in file_sets[rqstd_file]:
                srv_lst_for_this_request_excld_incom_srv.remove(incoming_srv)
#            print(srv_lst_for_this_request_excld_incom_srv)
            if len(srv_lst_for_this_request_excld_incom_srv) > 0:
                dmin = 2*srv_num # some large number, eg, larger that the graph diameter!
                for nd in srv_lst_for_this_request_excld_incom_srv:
                    #d = nx.shortest_path_length(G, source=incoming_srv, target=nd)
                    #dtmp = all_sh_path_len_G[nd]
                    dtmp = shortest_path_matrix[incoming_srv, nd]
                    if dtmp < dmin:
                        dmin = dtmp
                        nearest_neighbor = nd
#            print(file_sets[rqstd_file])
#            print(srv_lst_for_this_request_excld_incom_srv)

            if incoming_srv in file_sets[rqstd_file]:
                srv0 = incoming_srv
            else:
                srv0 = nearest_neighbor

            # Up to this point, we set srv1 also to be the same as srv0
            srv1 = srv0

            twochoice_search_space = []
            if len(file_sets[rqstd_file]) > 1:
                # Now we have to find a subset of nodes that have the requested file and
                # are within a distant "(1+alpha)*dmin" from the requesting server.
                for nd in file_sets[rqstd_file]:
                    if shortest_path_matrix[incoming_srv, nd] <= (1+alpha)*dmin and nd != srv0:
                        twochoice_search_space.append(nd)

                if len(twochoice_search_space) > 0:
                    while srv1 == srv0:
                        srv1 = twochoice_search_space[np.random.randint(len(twochoice_search_space))]
                        #srv1 = file_sets[rqstd_file][np.random.randint(len(file_sets[rqstd_file]))]

            #if len(file_sets[rqstd_file]) > 1:
            #    while srv1 == srv0:
            #        srv1 = file_sets[rqstd_file][np.random.randint(len(file_sets[rqstd_file]))]

            # Implement power of two choices
#            print(srv0)
            load0 = srvs[srv0].get_load()
            load1 = srvs[srv1].get_load()
            if load0 > load1:
                srvs[srv1].add_load()
                #total_cost += nx.shortest_path_length(G, source=incoming_srv, target=srv1)
                #total_cost += all_sh_path_len_G[srv1]
                total_cost += shortest_path_matrix[incoming_srv, srv1]
            elif load0 < load1:
                srvs[srv0].add_load()
                if srv0 != incoming_srv:
                    #total_cost += nx.shortest_path_length(G, source=incoming_srv, target=srv0)
                    #total_cost += all_sh_path_len_G[srv0]
                    total_cost += shortest_path_matrix[incoming_srv, srv0]
            elif np.random.randint(2) == 0:
                srvs[srv0].add_load()
                if srv0 != incoming_srv:
                    #total_cost += nx.shortest_path_length(G, source=incoming_srv, target=srv0)
                    #total_cost += all_sh_path_len_G[srv0]
                    total_cost += shortest_path_matrix[incoming_srv, srv0]
            else:
                srvs[srv1].add_load()
                #total_cost += nx.shortest_path_length(G, source=incoming_srv, target=srv1)
                #total_cost += all_sh_path_len_G[srv1]
                total_cost += shortest_path_matrix[incoming_srv, srv1]
        else:
            outage_num +=  1


    print('The "Trade-off" simulator is done.')
#    print('------------------------------')

    # At the end of simulation, find maximum load, etc.
    loads = [srvs[i].get_load() for i in range(srv_num)]
    maxload = max(loads)
    avgcost = total_cost / srv_num
    outage = outage_num / srv_num

#    print(loads)
#    print(maxload)
#    print(total_cost)

    return {'loads': loads, 'maxload': maxload, 'avgcost': avgcost, 'outage': outage}


#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
# This function randomly places library files into the server caches.
#
# Input:
# srv_num = number of servers
# file_num = number of files in the library
# cache_sz = size of cache in each server
# chnk_num = number of chunks each file is divided to
# placement_dist = the file popularity distribution
#
# Return:
#
def srv_cache_placement(srv_num, file_num, cache_sz, chnk_num, placement_dist, place_dist_param, srvs):


    # List of sets of servers containing a chunk from each file. The list is enumerate by the
    # file index and for each file we have a list that contains the servers cached a chunk from that file.
    file_sets = [[] for i in range(file_num)]

    # List of all cached files at the end of placement process
    list_cached_files = []

    # The index of chunk created (used) so far for each file
    chnk_used_so_far = [ -1 for i in range(file_num) ]

    if placement_dist == 'Uniform':
        # Randomly places cache_sz files in each servers
        print('Start randomly placing {} files in each server with Uniform dist.'.format(cache_sz))

        # First put randomly one chunk of each file in a subset of servers (assuming file_num =< srv_num * cache size)
        all_chunks = [ [fl, chnk] for fl in range(file_num) for chnk in range(chnk_num) ]
        print(all_chunks)
        all_chunks = np.random.permutation(all_chunks).tolist()
        print(all_chunks)
        #lst = np.random.permutation(srv_num)[0:file_num]
        for i in range(file_num * chnk_num):
            chnk_used_so_far[all_chunks[i][0]] += 1
            srv_indx = i % srv_num
            file_sets[all_chunks[i][0]].append(srv_indx)
            srvs[srv_indx].append_files_list(all_chunks[i])

        print(chnk_used_so_far)
        #for i, s in enumerate(lst):
        #    file_sets[i].append(s)
        #    srvs[s].set_files_list([i])
        #    list_cached_files.append(i)

        # Then fills the rest of empty cache places
        for s in np.arange(srv_num, dtype=np.int32):
            #print(s)
            srvlst = srvs[s].get_files_list()
            print(srvlst)
            #tmp_lst = np.random.permutation(file_num)[0:cache_sz - len(srvlst)]
            tmp_lst = np.random.randint(0, file_num, cache_sz * chnk_num - len(srvlst))
            #srvs[s].set_files_list(srvlst.extend(tmp_lst))
            for j in range(len(tmp_lst)):
                chnk_used_so_far[tmp_lst[j]] += 1
                srvs[s].append_files_list([tmp_lst[j], chnk_used_so_far[tmp_lst[j]]])
                if s not in file_sets[tmp_lst[j]]:
                    file_sets[tmp_lst[j]].append(s)

        # Fill the list of cached files
        #for i in range(file_num):
        #    if chnk_used_so_far >= chnk_num - 1:
        #        list_cached_files.append(i)

        print('Done with randomly placing {} files in each server with Uniform dist.'.format(cache_sz))
        #print(chnk_used_so_far)
        #for s in range(srv_num):
        #    print(srvs[s].get_files_list())
    elif placement_dist == 'Zipf':
        # Randomly places 'cache_sz * chnk_num' chunks in each servers with Zipf dist.
        gamma = place_dist_param['gamma']  # Parameter of Zipf distribution.
        print('Start randomly placing {} chunks in each server according to Zipf dist. with gamma={}'.\
              format(cache_sz*chnk_num, gamma))

        for s in np.arange(srv_num, dtype=np.int32):
            # Generates some sample from a zipf distribution over the set {1,...,file_num}
            tmp_zipf_smpl = bounded_zipf(file_num, gamma, cache_sz*chnk_num)
            for j in np.arange(len(tmp_zipf_smpl), dtype=np.int32):
                chnk_used_so_far[tmp_zipf_smpl[j]] += 1
                srvs[s].append_files_list([ tmp_zipf_smpl[j], chnk_used_so_far[tmp_zipf_smpl[j]] ])
                if s not in file_sets[tmp_zipf_smpl[j]]:
                    file_sets[tmp_zipf_smpl[j]].append(s)

        #    srvs[s].set_files_list(tmp_lst)
        #    for j in np.arange(len(tmp_lst), dtype=np.int32):
        #        if s not in file_sets[tmp_lst[j]]:
        #            file_sets[tmp_lst[j]].append(s)
        #        if tmp_lst[j] not in list_cached_files:
        #            list_cached_files.append(tmp_lst[j])
        #print(list_chached_files)
        print('Done with randomly placing {} files in each server with Zipf dist.'.format(cache_sz))

    # Fill the list of cached files
    for i in range(file_num):
        if chnk_used_so_far[i] >= chnk_num - 1:
            list_cached_files.append(i)


    for s in range(srv_num):
        print(srvs[s].get_files_list())

    print(list_cached_files)
    print(len(list_cached_files))

    print(chnk_used_so_far)
    print(len(chnk_used_so_far))
    print(sum(chnk_used_so_far))


    return srvs, file_sets, list_cached_files
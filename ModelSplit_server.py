from Common.Server.fl_grpc_server import FlGrpcServer
from Common.Grpc.fl_grpc_pb2 import GradResponse_float
from Common.Handler.handler import Handler
import Common.config as config
import torch
import hdbscan
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_distances
from scipy import stats
import numpy as np
from Common.Model.LeNet import LeNet
from crab.metrics.pairwise import adjusted_cosine
#from mpc_clustering.mpc_dbscan import *
import time
import subprocess
from crypten.mpc import MPCTensor
import crypten
import crypten.mpc as mpc


class ClearFLGuardServer(FlGrpcServer):
    def __init__(self, address, port, config, handler, num_model_params):
        super(ClearFLGuardServer, self).__init__(config=config)
        self.address = address
        self.port = port
        self.config = config
        self.handler = handler
        self.num_model_params = num_model_params

    def UpdateGrad_float(self, request, context): # using flguard
        data_dict = {request.id: request.grad_ori}
        #print("have received:", data_dict.keys())
        #rst,#id = super().process(dict_data=data_dict, handler=self.handler.computation)
        #if
        rst, split_label = super().process(dict_data=data_dict, handler=self.handler.computation)

        num_groups = len(split_label)
        for i in range(num_groups):
            if request.id in split_label[i]:
               return GradResponse_float(grad_upd=rst[i*self.num_model_params:(i+1)*self.num_model_params])


class FLGuardGradientHandler(Handler):
    def __init__(self, num_workers, f, weights):
        super(FLGuardGradientHandler, self).__init__()
        self.num_workers = num_workers
        self.f = f
        self.weights = weights
        self.lambdaa = 0.001


    def computation(self, data_in):
        # cluster
        weights_in = np.array(data_in).reshape((self.num_workers, -1))
        tic_cluster = time.time()
        ##################################################################
        np.savetxt('Player-Data/Input-P0-0', weights_in, fmt='%s', delimiter=' ')
        UNCLASSIFIED = False
        NOISE = -1

        def _region_query(point_id, eps, n_points, distance_enc):
            seeds = [point_id]
            for i in range(n_points):
                if i == point_id:
                    continue
                if (distance_enc[point_id, i] == 1):
                    seeds.append(i)
            return seeds

        def _expand_cluster(classifications, point_id, cluster_id, eps, min_points, n_points, distance_enc):
            seeds = _region_query(point_id, eps, n_points, distance_enc)
            if len(seeds) < min_points:
                classifications[point_id] = NOISE
                return False
            else:
                classifications[point_id] = cluster_id
                for seed_id in seeds:
                    classifications[seed_id] = cluster_id

                while len(seeds) > 0:
                    current_point = seeds[0]
                    results = _region_query(current_point, eps, n_points, distance_enc)
                    if len(results) >= min_points:
                        for i in range(0, len(results)):
                            result_point = results[i]
                            if classifications[result_point] == UNCLASSIFIED or \
                                    classifications[result_point] == NOISE:
                                if classifications[result_point] == UNCLASSIFIED:
                                    seeds.append(result_point)
                                classifications[result_point] = cluster_id
                    seeds = seeds[1:]
                return True

        def dbscan(m, eps, min_points):
            cluster_id = 1
            n_points = dismension
            classifications = [UNCLASSIFIED] * n_points

            for point_id in range(0, n_points):
                if (classifications[point_id] == UNCLASSIFIED):
                    if (_expand_cluster(classifications, point_id, cluster_id, eps, min_points, n_points, m)):
                        cluster_id = cluster_id + 1

            return classifications
        tic_subpro = time.time()
        subprocess.run(['Scripts/ring.sh', 'distance -OF Player-Data/'])
        toc_subpro = time.time()
        print('subprocess time: ',toc_subpro - tic_subpro)
        distance = np.loadtxt('Player-Data/-P0-0', dtype=np.str_, encoding='utf-8')
        dismension = 10
        distance_temp = []
        for i in distance:
            i = i.strip('[[')
            i = i.strip('[')
            i = i.strip(']]')
            i = i.strip(']')
            i = i.strip(',')
            i = i.strip('],')
            distance_temp.append(int(i))

        distance_temp = np.array(distance_temp)
        edis = distance_temp.reshape((dismension, dismension))

        labels = dbscan(edis, eps=0.5, min_points=2)
        toc_cluster = time.time()
        print('clustering time:', toc_cluster-tic_cluster)
        ##################################################################
        # weights_in_average = np.mean(weights_in,axis=0)
        # distance_matrix = 1 - adjusted_cosine(weights_in,weights_in,weights_in_average)

        # distance_matrix = pairwise_distances(weights_in, metric='cosine')
        # distance_matrix = np.round(distance_matrix)
        # self.cluster.fit(distance_matrix)
        # label = self.cluster.labels_
        #

        label = np.array(labels)
        split_label = []
        if (label == -1).all():
            split_label = [[i for i in range(self.num_workers)]]
        else:
            label_elements = np.unique(label)
            for i in label_elements.tolist():
                split_label.append(np.where(label == i)[0].tolist())
            #b = np.where(label == 0)[0].tolist()
        # euclidean distance between self.weights and clients' weights
        weight_agg = []
        print(split_label)
        for b in split_label:
             weight_agg.append(np.mean(weights_in[b], axis=0))

        # #self.weights = weight_agg
        #
        weight_agg = np.array(weight_agg).flatten()

        return weight_agg, split_label

if __name__ == "__main__":
    PATH = './Model/LeNet'
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    model.load_state_dict(torch.load(PATH))
    weights = []
    for param in model.parameters():
        weights += param.data.view(-1).numpy().tolist()
    gradient_handler = FLGuardGradientHandler(num_workers=config.num_workers, f = config.f, weights=np.array(weights))

    num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flguard_server = ClearFLGuardServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler, num_model_params=num_model_params)
    flguard_server.start()

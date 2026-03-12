import os
import torch
import loaddatas as lds
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import SIMBLOCKGNN as SIMGNN
from sklearn.metrics import roc_auc_score,average_precision_score
from torch.nn.init import xavier_normal_ as xavier
from criminal_nets_reader import get_criminal_net_edge_index, get_criminal_net_node_features, get_phonecalls_net_edges_index

path = os.getcwd()

def train():
    model.train()
    optimizer.zero_grad()
    emb = model.g_encode(data)
    x, y = model.s_encode(data, emb) # emb from encode's, i.e., Gconv's output
    loss = F.binary_cross_entropy(x,y)
    loss.backward()
    optimizer.step()
    return x

def test():
    model.eval()
    accs = []
    emb = model.g_encode(data)
    for type in ["val", "test"]:
        pred,y = model.s_encode(data,emb,type=type)
        pred,y = pred.cpu(),y.cpu()
        if type == "val":
            accs.append(F.binary_cross_entropy(pred, y))
            pred = pred.data.numpy()
            roc = roc_auc_score(y, pred)
            accs.append(roc)
            acc = average_precision_score(y,pred)
            accs.append(acc)
        else:
            pred = pred.data.numpy()
            roc = roc_auc_score(y, pred)
            accs.append(roc)
            acc = average_precision_score(y, pred)
            accs.append(acc)
    return accs

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        xavier(m.weight)
        if not m.bias is None:
            torch.nn.init.constant_(m.bias, 0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


#d_names = ['PPI']; times=range(20)
#d_names = ["Cora", "Photo", "PubMed", "Computers"]
d_names = ["Montagna_phonecalls"] # Montagna_phonecalls
meetings_edgelist = pd.read_csv(path + '/criminal_nets/Montagna_meetings_edgelist.csv').values
phonecalls_edgelist = pd.read_csv(path + '/criminal_nets/Montagna_phonecalls_edgelist.csv').values
times=range(1)


wait_total= 100
total_epochs = 5000


pipelines=['SIMGNN'] # where pipelines was defined!
pipeline_acc={'SIMGNN':[i for i in times]}
pipeline_acc_sum={'SIMGNN':0}
pipeline_roc={'SIMGNN':[i for i in times]}
pipeline_roc_sum={'SIMGNN':0}
pipeline_acc_same={'SIMGNN':[i for i in times]}
pipeline_acc_same_sum={'SIMGNN':0}
pipeline_roc_same={'SIMGNN':[i for i in times]}
pipeline_roc_same_sum={'SIMGNN':0}
pipeline_acc_diff={'SIMGNN':[i for i in times]}
pipeline_acc_diff_sum={'SIMGNN':0}
pipeline_roc_diff={'SIMGNN':[i for i in times]}
pipeline_roc_diff_sum={'SIMGNN':0}


for d_name in d_names:
    #f2 = open('scores/pipe_benchmark_' + d_name + '_LP_scores.txt', 'w+')
    #f2.write('{0:7} {1:7}\n'.format(d_name, 'SIMGNN'))
    #f2.flush()
    for data_cnt in times:
        for Conv_method in pipelines: # where Conv_method is SIMGNN
            if d_name in ['Rand_nnodes_github1000', 'PPI']:
                data = dataset[data_cnt]
            else:
                phonecalls_data_ = Data(edge_index= get_criminal_net_edge_index(phonecalls_edgelist), label = np.unique(phonecalls_edgelist[:, :2]))
                dataset = Data(name='Montagna_phonecalls',
                               x=get_criminal_net_node_features(data_type="phonecalls_data", data=phonecalls_data_),
                               edge_index=get_phonecalls_net_edges_index(phonecalls_data_),
                               label=np.unique(phonecalls_edgelist[:, :2]), num_classes=None)  # lds.loaddatas(d_name)
                data = dataset#dataset[0]
            if d_name in ['Rand_nnodes_github1000']:
                data.x = data.x[:, :10]
            #data.x = torch.ones(data.x.size())
            #index = [i for i in range(len(data.y))]
            if d_name != "PPI":
                model, data = locals()[Conv_method].call(data, dataset.name, data.x.size(1), dataset.num_classes)
            else:
                model, data = locals()[Conv_method].call(data, 'PPI', data.x.size(1), dataset.num_classes)
            model.apply(weights_init)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0) #0.001
            best_val_acc = test_acc_same = test_acc_diff = test_acc = 0.0
            best_val_roc = test_roc_same = test_roc_diff = test_roc = 0.0
            best_val_loss = np.inf
            # train and val/test
            wait_step = 0
            print(data_cnt)

            # train and test
            for epoch in range(1, total_epochs + 1):
                print(epoch)
                pred = train()
                val_loss, val_roc, val_acc, tmp_test_roc, tmp_test_acc = test()
                if val_roc >= best_val_roc:
                    test_acc = tmp_test_acc
                    test_roc = tmp_test_roc
                    best_val_acc = val_acc
                    best_val_roc = val_roc
                    best_val_loss = val_loss
                    wait_step = 0
                else:
                    wait_step += 1
                    if wait_step == wait_total:
                        print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc,
                              ', Max roc: ', best_val_roc)
                        break
                print(best_val_roc)
            del model
            del data
            # print result

            pipeline_acc[Conv_method][data_cnt] = test_acc
            pipeline_roc[Conv_method][data_cnt] = test_roc
            print("final acc. result is:", test_roc)

            log = 'Epoch: ' + str(
                total_epochs) + ', dataset name: ' + d_name + ', Method: ' + Conv_method + ' Test pr: {:.4f}, roc: {:.4f} \n'
            print((log.format(pipeline_acc[Conv_method][data_cnt], pipeline_roc[Conv_method][data_cnt])))
            #print(pred)

            #f2.write('{}, {:.4f}, {:.4f}\n'.format(data_cnt, pipeline_acc[Conv_method][data_cnt],
            #                                         pipeline_roc[Conv_method][data_cnt],))
            #f2.flush()
    #f2.write('{0:4} {1:4f}\n'.format('std', np.std(pipeline_acc[Conv_method])))
    #f2.write('{0:4} {1:4f}\n'.format('mean', np.mean(pipeline_acc[Conv_method])))
    #f2.write('{0:4} {1:4f}\n'.format('std', np.std(pipeline_roc[Conv_method])))
    #f2.write('{0:4} {1:4f}\n'.format('mean', np.mean(pipeline_roc[Conv_method])))
    #f2.close()

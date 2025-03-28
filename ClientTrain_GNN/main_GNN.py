import copy
import itertools
from random import shuffle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ClientTrain.utils.options import args_parser
from ClientTrain.models.Update import LocalUpdate,DatasetSplit
from ClientTrain.LongLifeMethod.EWC import Appr as EWC_Appr
from ClientTrain.LongLifeMethod.EWC import LongLifeTrain as EWC_LongLifeTrain
from ClientTrain.LongLifeMethod.MAS import Appr as MAS_Appr
from ClientTrain.LongLifeMethod.MAS import LongLifeTrain as MAS_LongLifeTrain
from ClientTrainGNN.LongLifeMethod.HeTa_GEM import Appr as GEM_Appr
from ClientTrainGNN.LongLifeMethod.HeTa_GEM import LongLifeTrain as GEM_LongLifeTrain
from ClientTrainGNN.LongLifeMethod.HeTa_GEM import LongLifeTest as GEM_LongLifeTest
from torch.utils.data import DataLoader
import time
from Agg.FedDag import FedDag
import copy
from ClientTrainGNN.models.GCN import SimpleGCN
from Agg.AggModel.gcn import SimpleGCN as KDmodel
from ClientTrainGNN.dataset.MiniGC import MiniGCTask
from Agg.OTFusion.cifar.models.gcn import GCN
import dgl
def collate(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)
if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.dataset = 'MiniGC'
    lens = np.ones(args.num_users)

    dataset = MiniGCTask(args.task, args.num_users)
    dataset_train = dataset.get_clients_train()
    dataset_test = dataset.get_clients_test()
    # dict_users_test = [copy.deepcopy(dict_users_test) for i in range(2) for dict_user in dict_users_test]

    print(args.alg)
    write = SummaryWriter('/data/lpyx/FedAgg/ClientTrainGNN/log/MiniGC/FedHeTA/'+args.clmethod+'/server_epoch5_high20_dag_' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac))
    # build model
    # net_glob = get_model(args)
    clients_models = []
    hidden = 4
    for i in range(args.num_users):
        if i % 2 == 0:
            hidden *= 2
        clients_models.append(SimpleGCN(1, hidden_dim = hidden, n_classes=8 * args.task))

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    task=-1
    kd_model = KDmodel(1, 8, 8)
    if args.clmethod == 'EWC':
        apprs = [EWC_Appr(clients_models[i].to(args.device), None,lr=args.lr, nepochs=args.local_ep, args=args,kd_model=kd_model) for i in range(args.num_users)]
    elif args.clmethod == 'MAS':
        apprs = [MAS_Appr(clients_models[i].to(args.device), None, lr=args.lr, nepochs=args.local_ep, args=args,
                      kd_model=kd_model) for i in range(args.num_users)]
    elif args.clmethod == 'GEM':
        apprs = [GEM_Appr(clients_models[i].to(args.device), kd_model, 3 * 32 * 32, 100, 10, args) for i in range(args.num_users)]

    print(args.round)
    serverAgg = FedDag(int(args.frac * args.num_users),int(args.frac * args.num_users * 2),datasize=[3,32,32],dataname='MiniGC', model=GCN(1,8,8))
    w_globals = []
    for iter in range(args.epochs):
        if iter % (args.round) == 0:
            task+=1
            w_globals = []
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        if iter % (args.round) == args.round - 1:
            print("*"*100)
            print("Last Train")
            idxs_users = [i for i in range(args.num_users)]
        else:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        times_in = []
        total_len = 0
        tr_dataloaders= None
        all_kd_models = []

        for ind, idx in enumerate(idxs_users):
            start_in = time.time()

            tr_dataloaders = DataLoader(dataset_train[idx][task], batch_size=args.local_bs,shuffle=True, collate_fn=collate)
                # if args.epochs == iter:
                #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_ft])
                # else:
                #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_tr])

                # appr = Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name)


            appr = apprs[idx]
            if len(w_globals) != 0:
                agg_client = [w['client'] for w in w_globals]
                if idx in agg_client:
                    cur_state = appr.cur_kd.state_dict()
                    for k in cur_state.keys():
                        if k in w_globals[agg_client.index(idx)]['model']:
                            cur_state[k] = w_globals[agg_client.index(idx)]['model'][k]
                    appr.cur_kd.load_state_dict(cur_state)


            # appr.set_model(net_local.to(args.device))
            appr.set_trData(tr_dataloaders)
            last = iter == args.epochs

            if args.clmethod == 'EWC':
                kd_models,loss, indd = EWC_LongLifeTrain(args,appr,iter,None,idx)
            elif args.clmethod == 'MAS':
                kd_models, loss, indd = MAS_LongLifeTrain(args, appr, iter, None, idx)
            elif args.clmethod == 'GEM':
                kd_models, loss, indd = GEM_LongLifeTrain(args, appr, tr_dataloaders, iter, idx)


            all_kd_models.append({'client':idx, 'models': kd_models})

            loss_locals.append(copy.deepcopy(loss))
        if iter % args.round == args.round - 1:
            w_globals = []
        else:
            w_globals = serverAgg.update(all_kd_models,task)


        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        loss_test, acc_test = 0, 0
        for i in range(args.num_users):
            l,acc = GEM_LongLifeTest(apprs[i],task, dataset_test[i][:task+1])
            loss_test += l
            acc_test += acc
        acc_test /= args.num_users
        loss_test /= args.num_users

        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
        write.add_scalar('task_finish_and _agg', acc_test, iter)
        if iter % (args.round) == args.round - 1:
            for i in range(args.num_users):
                tr_dataloaders = DataLoader(dataset_train[i][task], batch_size=args.local_bs,shuffle=True, collate_fn=collate)
                client_state = apprs[i].prune_kd_model(tr_dataloaders,task)
                if args.clmethod == 'GEM':
                    apprs[i].add_his_task(tr_dataloaders)
                serverAgg.add_history(client_state)

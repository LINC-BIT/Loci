import copy
import itertools
from random import shuffle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ClientTrain.utils.options import args_parser
from ClientTrain.utils.train_utils import get_data, get_model, read_data
from ClientTrain.models.Update import LocalUpdate,DatasetSplit
from ClientTrain.models.test import test_img_local_all
from ClientTrainFedGen.LongLifeMethod.GenClient import UserpFedGen as Appr,LongLifeTrain
from ClientTrainFedGen.AggModel.sixcnn import SixCNN
from torch.utils.data import DataLoader
from Agg.FedGen import FedGen
import time
from Agg.FedDag import FedDag
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])
    client_task=[[j for j in range(args.task)] for i in range(args.num_users)]
    for i in client_task:
        shuffle(i)

    print(args.alg)
    write = SummaryWriter('./log/ganbase/ganbase_high20_dag_' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac))
    # build model
    # net_glob = get_model(args)
    net_glob = SixCNN([3,32,32],outputsize=100)
    net_glob.train()

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)

    if args.alg == 'fedavg' or args.alg == 'prox':
        w_glob_keys = []

    print(total_num_layers)
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    start = time.time()
    task=-1
    genserver = FedGen(args,net_glob)
    apprs = [Appr(copy.deepcopy(net_glob).to(args.device),genserver.generative_model,[i for i in range(10)], args=args) for i in range(args.num_users)]
    print(args.round)

    for iter in range(args.epochs):
        if iter % (args.round) == 0:
            task+=1
        loss_locals = []
        clients_parameters = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        if iter % (args.round) == args.round - 1:
            print("*"*100)
            print("Last Train")
            idxs_users = [i for i in range(args.num_users)]
        else:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        tr_dataloaders= None
        clients_label_counts=[]
        for ind, idx in enumerate(idxs_users):
            start_in = time.time()

            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx][:args.m_ft],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            appr = apprs[idx]

            # appr.set_model(net_local.to(args.device))
            appr.set_trData(tr_dataloaders)
            last = iter == args.epochs
            client_parameters,loss, label_counts, indd = LongLifeTrain(args,appr,iter,None,idx)
            clients_label_counts.append(label_counts)
            clients_parameters.append(client_parameters)

            loss_locals.append(copy.deepcopy(loss))
        server_parameters = genserver.aggregate(task,clients_parameters,clients_label_counts)

        if iter % args.round == args.round - 1:
            w_globals = []


        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)


        acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test,task,apprs=apprs,w_locals=None,return_all=False,write=write,round=iter,client_task=client_task)
        accs.append(acc_test)

        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
        if iter % args.round != args.round - 1:
            for appr in apprs:
                appr.model.load_state_dict(server_parameters)
        #
        # if iter >= args.epochs - 10 and iter != args.epochs:
        #     accs10 += acc_test / 10
        # if iter >= args.epochs - 10 and iter != args.epochs:
        #     accs10_glob += acc_test / 10

            # model_save_path = './save/Baseline/0.4/accs_Fedavg_lambda_'+str(args.lamb) +str('_') + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
            #     args.shard_per_user) + '_iter' + str(iter) + '_frac_'+str(args.frac)+'.pt'
            # torch.save(net_glob.state_dict(), model_save_path)

    # print('Average accuracy final 10 rounds: {}'.format(accs10))
    # if args.alg == 'fedavg' or args.alg == 'prox':
    #     print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))

from dgl.data import MiniGCDataset
from random import shuffle
from torch.utils.data import DataLoader
import torch
import dgl
def collate(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)
class MiniGCTask:
    def __init__(self, tasknum=10, clientnum=10):
        self.task_num = tasknum
        self.client_num = clientnum
        self.clients_task_train = []
        self.clients_task_test = []
        client_taks_id = [[j for j in range(tasknum)] for i in range(clientnum)]
        for i in client_taks_id:
            shuffle(i)
        for c in range(clientnum):
            cur_client_train = []
            cur_client_test = []
            for t in range(tasknum):
                cur_client_train.append(MiniGCDataset(1000, 10 + t * 15, 25 + t * 15))
                cur_client_test.append(MiniGCDataset(400, 10 + t * 15, 25 + t * 15))
            cur_client_id = client_taks_id[c]
            sort_train = sorted(cur_client_train, key=lambda x: cur_client_id[cur_client_train.index(x)])
            sort_test = sorted(cur_client_test, key=lambda x: cur_client_id[cur_client_test.index(x)])
            self.clients_task_train.append(sort_train)
            self.clients_task_test.append(sort_test)

    def get_clients_train(self):
        return self.clients_task_train

    def get_clients_test(self):
        return self.clients_task_test

    def get_clients_train_dataloader(self):
        train_clients_loaders = []
        for c in range(self.client_num):
            for t in range(self.task_num):
                DataLoader(self.clients_task_train[c][t],batch_size=100,shuffle=True,collate_fn=collate)


dataset = MiniGCTask(5, 5)
train = dataset.get_clients_train()
test = dataset.get_clients_test()
print('1')

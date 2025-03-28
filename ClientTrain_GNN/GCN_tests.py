from dgl.data import MiniGCDataset, TUDataset
import dgl
import torch
from dgl.nn.pytorch import GraphConv, GATConv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def collate(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)


class PairNorm(nn.Module):
    def __init__(self, mode='PN-SCS', scale=1.0):
        """
            mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version

            ('SCS'-mode is not in the paper but we found it works well in practice,
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.gcn1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
        self.gcn2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积
        self.fc = nn.Linear(hidden_dim, n_classes)  # 定义分类器

    def forward(self, g):
        """g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量
        """
        # 我们用节点的度作为初始节点特征。对于无向图，入度 = 出度
        h = g.in_degrees().view(-1, 1).float()  # [N, 1]
        # 执行图卷积和激活函数
        h = F.relu(self.gcn1(g, h))  # [N, hidden_dim]
        h = F.relu(self.gcn2(g, h))  # [N, hidden_dim]
        g.ndata['h'] = h  # 将特征赋予到图的节点
        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')  # [n, hidden_dim]
        return self.fc(hg)  # [n, n_classes]


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv((in_dim, in_dim), num_hidden, heads[0], feat_drop, attn_drop, negative_slope))
        self.norm_layers.append(PairNorm())
        self.n_classes = num_classes
        self.n_hidden = num_hidden
        self.heads = heads
        self.n_known = 5

        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
            self.norm_layers.append(PairNorm())

        self.gat_layers.append(GATConv((num_hidden * heads[-2], num_hidden * heads[-2]), num_classes, heads[-1],
                                       feat_drop, attn_drop, negative_slope, residual))

    def forward(self, block):
        h = block.in_degrees().view(-1, 1).float()  # [N, 1]
        '''
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
            h = self.activation(h)
        logits = self.gat_layers[-1](g, h).mean(1)

        '''
        for l, layer in enumerate(self.gat_layers):

            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)

            h_dst = h[:block.number_of_dst_nodes()]  # torch.Size([9687, 602])
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))

            if l != len(self.gat_layers) - 1:
                h = h.flatten(1)
                h = self.activation(h)
            # h = self.dropout(h)
            else:
                h = h.mean(1)
        block.ndata['h'] = h
        hg = dgl.mean_nodes(block, 'h')
        return hg

    def inference(self, g, x, batch_size):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = torch.arange(g.number_of_nodes())
        for l, layer in enumerate(self.gat_layers):
            y = torch.zeros(g.number_of_nodes(),
                            (self.heads[l] * self.n_hidden) if l != len(self.gat_layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                device = torch.device("cuda:0")
                block = block.to(device)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].cuda()
                h_dst = h[:block.number_of_dst_nodes()]
                h, _ = layer(block, (h, h_dst))
                if l != len(self.gat_layers) - 1:
                    h = h.flatten(1)
                    h = self.activation(h)
                else:
                    h = h.mean(1)

                y[start:end] = h.cpu()

            x = y
        return y
def train(model, data_loader):
    epoch_losses = []
    loss_func = nn.CrossEntropyLoss()
    # 定义Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(200):
        model.train()
        epoch_loss = 0
        for iter, (batchg, label) in enumerate(data_loader):
            batchg = batchg.to(device)
            label = label.to(device)
            prediction = model(batchg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)
        if epoch % 9 == 0:
            eval(model, data_loader)
            torch.save(model.state_dict(), 'test_models/GCN_{0}.pth'.format(str(epoch)))

def eval(model, testset):
    from sklearn.metrics import accuracy_score
    # model.load_state_dict(torch.load('GCN.pth'))
    # test_loader = DataLoader(testset, batch_size=1000, collate_fn=collate)
    model.eval()
    test_pred, test_label = [], []
    with torch.no_grad():
        for it, (batchg, label) in enumerate(testset):
            batchg = batchg.to(device)
            label = label.to(device)
            pred = torch.softmax(model(batchg), 1)
            pred = torch.max(pred, 1)[1].view(-1)
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += label.cpu().numpy().tolist()
    print("accuracy: ", accuracy_score(test_label, test_pred))

from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
# 数据集包含了80张图。每张图有10-20个节点
dataset = MiniGCDataset(80, 10, 15)
graph, label = dataset[15]
fig, ax = plt.subplots()
nx.draw(graph.to_networkx(), ax=ax)
ax.set_title('Class: {:d}'.format(label))
plt.savefig('class2.png')
# device = torch.device("cuda:1")
#
# trainset = MiniGCDataset(8000, 10, 20)
# testset1 = MiniGCDataset(800, 10, 20)
# # testset2 = MiniGCDataset(800, 20, 50)
# # testset3 = MiniGCDataset(800, 50, 100)
# # testset4 = MiniGCDataset(800, 100, 200)
# data_loader = DataLoader(trainset, batch_size=80, shuffle=False,
#                          collate_fn=collate)
#
# heads = [8, 1]
# model = Classifier(1, 8, 8).to(device)
# # model = GAT(1, 1, 8, 8, heads, F.relu, 0.6, 0.6, 0.2, False).to(device)
# train(model, data_loader)
# test_loader1 = DataLoader(testset1, batch_size=800, collate_fn=collate)
# # test_loader2 = DataLoader(testset2, batch_size=800, collate_fn=collate)
# # test_loader3 = DataLoader(testset3, batch_size=800, collate_fn=collate)
# # test_loader4 = DataLoader(testset4, batch_size=800, collate_fn=collate)
# eval(model, test_loader1)
# eval(model, test_loader2)
# eval(model, test_loader3)
# eval(model, test_loader4)
# model.train()
#
# torch.save(model.state_dict(), 'GCN.pth')
#
#
#
# # torch.save(model.state_dict(), 'GCN.pth')

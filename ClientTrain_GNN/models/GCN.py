import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(SimpleGCN, self).__init__()
        self.gcn1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
        self.gcn2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积
        self.last = nn.Linear(hidden_dim, n_classes)  # 定义分类器
        self.nc_per_task = 8
        self.outputsize = n_classes

    def forward(self, g, t=-1):
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
        output = self.last(hg)
        if t != -1:
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.outputsize:
                output[:, offset2:self.outputsize].data.fill_(-10e10)
        return output  # [n, n_classes]
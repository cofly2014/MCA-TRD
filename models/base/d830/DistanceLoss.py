import torch
import torch.nn as nn
from itertools import combinations


class DistanceLoss(nn.Module):
    "Compute the Query-class similarity on the patch-enriched features."

    def __init__(self, args, temporal_set_size=3):
        super(DistanceLoss, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.dropout = nn.Dropout(p=0.1)

        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)  # 对frame_idxs进行大小为temporal_set_size的组合
        # 把数据放在cuda上
        #self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples)  # 28 for tempset_2

        #
        #self.clsW = nn.Linear(self.args.trans_linear_in_dim * self.temporal_set_size,
        #                      self.args.trans_linear_in_dim // 2).cuda()
        self.clsW = nn.Linear(self.args.trans_linear_in_dim * self.temporal_set_size, self.args.trans_linear_in_dim // 2).cuda()
        self.relu = torch.nn.ReLU()

        # support_set: 样本数量，每个样本的frame数量，每个frame的表征； support_labels 对应样本数量；  queries：样本数量，每个样本的frame数量，每个frame的表征

    def forward(self, support_set, support_labels, queries):
        # support_set : 5 x 8 x 512, support_labels: [0,1,2,3,4], queries: 25 x 8 x 512
        n_queries = queries.shape[0]  # 20
        n_support = support_set.shape[0]  # 25

        # Add a dropout before creating tuples
        support_set = self.dropout(support_set)  # 5 x 8 x 512
        queries = self.dropout(queries)  # 25 x 8 x 512

        # construct new queries and support set made of tuples of images after pe
        # 在support_set 中选择下表为p中元素（p是一个组合）的元素，例如 选择之后  10 8 512--> 10, 2, 512, 然后把后面两维压缩-->10,1024, 循环遍历完毕是 28个 10,1024的张量
        #s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]  # in self.tuples 所有选择的frame的排列组合
        #q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        # s: 28个[10  1024] ;  q: 28个[25  1024]
        #support_set = torch.stack(s, dim=-2).cuda()  # 10 x 28 x 1024
        #queries = torch.stack(q, dim=-2)  # 25 x 28 x 1024
        support_labels = support_labels.cuda()
        unique_labels = torch.unique(support_labels)  # 5
        # 25*28   1024(2,512)  self.clsW一个线性变换操作queries.view(-1, self.args.trans_linear_in_dim*self.temporal_set_size) 为 700，1024； ---> 700 512
        #query_embed = self.clsW(queries.view(-1, self.args.trans_linear_in_dim * self.temporal_set_size))  # 560[20x28] x 1024
        query_embed = self.clsW(queries.view(-1, self.args.trans_linear_in_dim ))
        # Add relu after clsW     ,, 25*28, 256
        query_embed = self.relu(query_embed)  # 560 x 1024
        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        dist_all = torch.zeros(n_queries, self.args.way)  # 20 x 5
        for label_idx, c in enumerate(unique_labels):
            # Select keys corresponding to this class from the support set tuples
            # support_set 10  28  1024， 在第0维度上， 选出第self._extract_class_indices(support_labels, c) 个  ;选出support_set上的 某一类的 向量
            class_k = torch.index_select(support_set, 0, self._extract_class_indices(support_labels, c))  # [ 1 x 28 x 1024] 第一维是 support_set中属于c类的 样本数量
            # class_k 维度为 n 28  1024   ； support_set为 n  28  1014
            # Reshaping the selected keys
            #class_k = class_k.view(-1, self.args.trans_linear_in_dim * self.temporal_set_size)  # [x28  1024]
            class_k = class_k.view(-1, self.args.trans_linear_in_dim )
            # class_k为  28  1024
            # Get the support set projection from the current class
            support_embed = self.clsW(class_k.to(queries.device))  # 140[5 x 28] x1024
            # support_embed 为 [x28  256]
            # Add relu after clsW
            support_embed = self.relu(support_embed)  # 140 x 1024
            # query_embed：所有的query_embed数量，28 组合的数量，表征######support_embed： 一个support类下的样本数量， 28 组合的数量，表征######support_embed：
            # Calculate p-norm distance between the query embedding and the support set embedding torch.cdist高效计算大矩阵相似度
            distmat = torch.cdist(query_embed, support_embed)  # 560[20 x 28] x 140[28 x 5]
            # distmat第一维是query的样本个数，第二维度是c类的support_set样本数量
            # min_dist 每个元素是 700个query中 每个query 和support中一个类的样本相比较 得到的最小值，这样会有 700个结果。然后再reshape
            # comment by guofei, 这里是否可以修改为DTW算法？
            min_dist = distmat.min(dim=1)[0].reshape(n_queries, self.tuples_len)  # 20[5-way x 4-queries] x 28
            # distmat.min(dim=1)[0]是数值，distmat.min(dim=1)[1]是下标； distmat.min(dim=1)[0] 25*28行,一列，代表 query中每一个pair对support中 pair的最小距离；min_dist 为 25  28
            # distmat.min(dim=1)[0] 每个query对c类的所有support set中样本距离最小的那个的距离。
            # distmat.min(dim=1)[0].reshape(n_queries, self.tuples_len) 每个query对c类的所有support set中样本距离最小的那个的距离。用二维表示
            # min_dist 行代表某一个query样本， 类代表28个组合，  某个query在某个组合下和support对比的最小距离。 Average across the 28 tuples
            query_dist = min_dist.mean(dim=1)  # 20   #第一维度代表每个query,所以就是每个query中所有28个pair对一个c类的support_set的最小距离 求平均
            # 用其表示为每个query对c类support的最小距离
            # query_dist维度为25
            # Make it negative as this has to be reduced.
            distance = -1.0 * query_dist
            c_idx = c.long()
            dist_all[:, c_idx] = distance  # Insert into the required location.
        # dist_all是 25  5  每个元素代表每个query对每个support类的距离
        return_dict = {'logits': dist_all}

        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
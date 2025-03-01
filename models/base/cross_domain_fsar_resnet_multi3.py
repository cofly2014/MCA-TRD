import torch.nn as nn
from torch import einsum
import torch.nn.functional as F

import random
from einops import rearrange
import os
import torch
#from models.base.clip_fsar import load, tokenize
from utils import getcombinations, EuclideanDistance, Euclidean_Distance
import copy
from models.base.resNet import MyResNet
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from gensim.test.utils import datapath



class PreNormattention_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v), **kwargs) + q


class Transformer_v1(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte=0.05, mlp_dim=2048,
                 dropout_ffn=0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                    # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                    PreNormattention_qkv(dim,
                                         Attention_qkv(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte)),
                    FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                ]))

    def forward(self, q, k, v):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(q, k, v)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x, x, x)
                x = ff(x) + x
        return x


class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class Attention_qkv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        bk = k.shape[0]
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', b=bk, h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', b=bk, h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)  # [30, 8, 8, 5]

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


'''
子序列生成器
'''


class Subsequence_Generator:
    def __init__(self, full_seq_len, sub_seq_len, sub_seq_num):
        self.full_seq_len = full_seq_len
        self.sub_seq_len = sub_seq_len
        self.sub_seq_num = sub_seq_num

    def generate_subsequence_comb(self):
        # 新文章实验   取得self loss  cross loss存在问题
        # 确定选帧  C82
        combinations = torch.as_tensor(getcombinations(self.full_seq_len, self.sub_seq_len)).cuda()
        combinations, _ = torch.sort(combinations, dim=-1)
        random_index = torch.as_tensor(random.sample(range(combinations.__len__()), self.sub_seq_num)).cuda()
        # 得到 c82数据取随机4个，还是排序
        combinations = torch.index_select(combinations, 0, random_index)
        # 取出特征 reshape操作   进入context local
        return combinations

    def __call__(self, ):
        return self.generate_subsequence_comb()


####################################################################

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


# 定义文本向量化函数
def text_to_vector(text, word2vec_model):
    words = text.lower().split()
    word_vectors = [word2vec_model[word] for word in words if word in word2vec_model]
    if word_vectors:
        # 将NumPy数组转换为PyTorch张量，并计算平均向量
        return torch.mean(torch.stack([torch.from_numpy(vec) for vec in word_vectors]), dim=0)
    else:
        return torch.zeros(word2vec_model.vector_size)

def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(
                - cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(
                - cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(
                - cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(
                - cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


class CROSS_DOMAIN_FSAR(nn.Module):
    """
    OTAM with a CNN backbone.
    """

    def __init__(self, args):
        super(CROSS_DOMAIN_FSAR, self).__init__()
        self.argss = args

        ###################restnet backbone###########################################
        self.backbone = MyResNet(self.argss)
        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        ##############################################################################

        #######################时序网络定义##############################################

        self.mid_dim = 512
        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)

        self.student_encoder = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.argss.TRANSFORMER_DEPTH))
        self.inner_teacher = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.argss.TRANSFORMER_DEPTH))
        self.decoder = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.argss.TRANSFORMER_DEPTH))
        #########################end 时序网络定义##########################################

        ######################监督学习的线性分类器###########################################
        self.classification_layer = nn.Sequential(
            nn.LayerNorm(self.mid_dim),
            nn.Linear(self.mid_dim, self.argss.class_num)
        )
        ##################################################################################

        ##############################一个从student网络到teacher网络的特征映射################
        self.mappingNet = nn.Sequential(
            nn.Conv1d(self.mid_dim // 1, self.mid_dim // 1, kernel_size=3, padding=1, groups=1),
            nn.ReLU(True),
            # nn.Linear(self.mid_dim // 1, self.mid_dim // 1),
            nn.Conv1d(self.mid_dim // 1, self.mid_dim // 1, kernel_size=3, padding=1, groups=1),
            nn.ReLU(True),
        )
        self.mappingLinear_seq_len = nn.Linear(4, 8)
        self.mappingLinear_seq_num = nn.Linear(5, 1)
        ##############################end 一个从student网络到teacher网络的特征映射#############

        # 子序列组合生成器
        self.Subsequence_Generator = Subsequence_Generator(self.argss.seq_len, self.argss.sub_seq_len, self.argss.sub_seq_num)
        self.copyStudent = True
        self.start_cross = self.argss.start_cross

        self.aa = self.argss.aa

    '''
    计算一个全序列样本，和改样本子序列之间判别概率的KL散度
    '''

    def self_seq_loss(self, P_local, P_global, bs):
        self_loss = sum([-torch.sum(torch.mul(torch.log(P_local[index]), P_global[index])) for index in range(bs)])
        return self_loss

    '''
    计算一个全序列样本，和该样本所属的类，的其他子序列之间判别概率的KL散度
    '''

    def cross_seq_loss(self, P_local, P_global, inputs, bs):
        shuffle_P_local = []
        target_label_dict = {}
        for i, v in enumerate(inputs["target_labels"]):
            if target_label_dict.get(v.item()):
                target_label_dict[v.item()].append(i)
            else:
                target_label_dict[v.item()] = [i]
        for i, value in enumerate(inputs["target_labels"]):
            # 提取classlabel相同的，获取index，index不同继续抽
            while True:
                mid = random.choice(target_label_dict[value.item()])
                if int(i) != mid:
                    break
                else:
                    pass
            shuffle_P_local.append(P_local[(mid):((mid + 1))])

        shuffle_P_local = torch.concat(shuffle_P_local, dim=0)
        cross_loss = sum([-torch.sum(torch.mul(torch.log(shuffle_P_local[index]), P_global[index])) for index in range(bs)])
        return cross_loss


    def forward(self, inputs, iteration=1):  # 获得support support labels, query, support real class
        support_images, support_labels, target_images, support_real_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels']  # [200, 3, 224, 224] inputs["real_support_labels"]
        #######################################################################################################
        target_labels = inputs["target_labels"]
        target_domain_set = inputs['target_domain_set']
        #######################################################################################################
        # 将target融入到episode的模式中, 获得target 无标签数据的backbone特征
        # 获得源数据和目标数据的backbone的特征
        support_features = self.get_feats(support_images)
        query_features = self.get_feats(target_images)
        target_domain_features = self.get_target_domain_feat(target_domain_set)
        ###################################################################################################
        '''
        feature_classification_in = torch.cat([support_features, query_features], dim=0)
        feature_classification = self.fc_norm(feature_classification_in.mean(1))
        class_logits = self.classifier(feature_classification)
        '''
        ######################################################################################
        # student 网络,是核心主部件
        # 通过student网络获得source_domain 中support和query的full_sequence的特征，其中support_features_g是原型
        # 空间上进行数据增强
        if iteration/2 ==0:
            enhanced_support_features = self.enhance_source(support_features, target_domain_features)
            enhanced_target_features = self.enhance_source(query_features, target_domain_features)
        else:
            enhanced_support_features = support_features
            enhanced_target_features = query_features

        enhanced_support_features = support_features
        enhanced_target_features = query_features
        support_features_g_pro, support_features_g, query_features_g = self.text_eh_temporal_transformer(enhanced_support_features, enhanced_target_features, support_labels)

        #下面这行代码是不使用空间增强后的特征
        #support_features_g_pro, support_features_g, query_features_g = self.text_eh_temporal_transformer(support_features, query_features, support_labels)

        source_domain_features_g = torch.concat([support_features_g, query_features_g], dim=0)
        cum_dist_g = -self.otam_distance(support_features_g_pro, query_features_g)  # 全局对比全局
        P_global = F.softmax(cum_dist_g, dim=-1)
        # 监督学习，对用backbone中提取的源域的数据进行分类
        class_logits = self.classification_layer(source_domain_features_g.mean(1))
        ##########################################################################################

        '''
        在训练阶段增加向inner teacher的知识蒸馏，向outter teacher的知识整理
        以及teacher网络权值的动量更新
        '''
        reconstruct_norm_distance = 0
        target_self_s_loss = 0
        self_loss = 0
        cross_loss = 0
        # 以target_domain_features为图库 来增强support的视频

        '''
        reconstruct_support_features = self.decoder(support_features_g, support_features_g, support_features_g)
        reconstruct_target_features = self.decoder(query_features_g, query_features_g, query_features_g)
        reconstruct_diff_support = support_features - reconstruct_support_features
        reconstruct_norm_sq_support = torch.norm(reconstruct_diff_support, dim=[-1]) ** 2
        reconstruct_norm_distance_support = torch.mean(reconstruct_norm_sq_support, dim=[0,1])  # 重建误差
        reconstruct_diff_target = query_features - reconstruct_target_features
        reconstruct_norm_sq_query = torch.norm(reconstruct_diff_target, dim=[-1]) ** 2
        reconstruct_norm_distance_query = torch.mean(reconstruct_norm_sq_query, dim=[0,1])   # 重建误差
        reconstruct_norm_distance = (reconstruct_norm_distance_support + reconstruct_norm_distance_query) / 2
        '''
        #target reconstruction损失
        target_domain_features_enc = self.student_encoder(target_domain_features, target_domain_features, target_domain_features)
        reconstruct_target_domain_features =self.decoder(target_domain_features_enc, target_domain_features_enc, target_domain_features_enc)
        reconstruct_diff_target_domain = reconstruct_target_domain_features - target_domain_features
        reconstruct_norm_target_domain = torch.norm(reconstruct_diff_target_domain, dim=[-2, -1]) ** 2 / (self.mid_dim)
        reconstruct_norm_distance_target_domain = torch.mean(reconstruct_norm_target_domain)  # 重建误差
        reconstruct_norm_distance = reconstruct_norm_distance + reconstruct_norm_distance_target_domain
        print("reconstruct_norm_distance: " + str(reconstruct_norm_distance))

        if self.training and iteration > self.start_cross:
        ########################################下面是global-local adapter#########################################
            if self.copyStudent:
                self.inner_teacher = copy.deepcopy(self.student_encoder)
                for param_t in self.inner_teacher.parameters():
                    param_t.requires_grad = False
                self.copyStudent = False
            print("start to do the knowledge distillation...")
            ##########################################################################################
            # student和inner_teacher之间
            query_features_comb_info = self.sub_seq_generate(query_features)
            support_features_comb_info = self.sub_seq_generate(support_features)
            cross_loss, self_loss, local_dist_g2l, local_dist_l2g = self.student2inner_teacher(cum_dist_g, target_labels, support_features_comb_info, support_features_g_pro, support_labels, query_features_comb_info, query_features_g)
            ##########################################################################################
            # 4. 通过课程学习 用student来动量更新两个teacher

            with torch.no_grad():
                m = 0.99980
                # 用student网络参数，动量跟新inner_teacher网络
                for param_q, param_k in zip(self.student_encoder.parameters(), self.inner_teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                # 用student网络参数，动量更新outter_teacher网络



        return_dict = {
            'class_logits': class_logits,
            'meta_logits': cum_dist_g,
            "reconstruct_distance": reconstruct_norm_distance,
            "self_logits": self_loss,
            "cross_logits": cross_loss
        }  # [5， 5] , [10 64]

        return return_dict

    def forward_test(self, inputs, iteration=1):  # 获得support support labels, query, support real class

        support_images, support_labels, target_images, support_real_class = inputs['support_set'], inputs[
            'support_labels'], inputs['target_set'], inputs['real_support_labels']  # [200, 3, 224, 224] inputs["real_support_labels"]

        #self.text_features_meta_testS = torch.stack(self.text_features_testS).cuda()
        #self.text_features_meta_testS = self.text_fc(self.text_features_meta_testS)
        #context_support = self.text_features_meta_testS[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
        #######################################################################################################
        target_labels = inputs["target_labels"]
        target_domain_set = inputs['target_domain_set']
        #######################################################################################################
        # 将target融入到episode的模式中, 获得target 无标签数据的backbone特征
        # 获得源数据和目标数据的backbone的特征
        support_features = self.get_feats(support_images)
        query_features = self.get_feats(target_images)
        ###################################################################################################
        # student 网络,是核心主部件
        # 通过student网络获得source_domain 中support和query的full_sequence的特征，其中support_features_g是原型
        support_features_g_pro, support_features_g, query_features_g = self.text_eh_temporal_transformer(support_features, query_features, support_labels)

        cum_dist_g = -self.otam_distance(support_features_g_pro, query_features_g)  # 全局对比全局
        return_dict = {
            'dis_logits': cum_dist_g
        }  # [5， 5] , [10 64]

        return return_dict

    def get_target_domain_feat(self, target_domain_images):
        target_domain_features = self.backbone(target_domain_images)[-1]  # self.backbone 为 clip的视觉分支visual ModifeidResnet 输出维度 [(way*shot*frames_number),1024]
        support_shape_ttm = target_domain_features.shape
        target_domain_features = target_domain_features.reshape(int(support_shape_ttm[0] / self.argss.seq_len), self.argss.seq_len,  support_shape_ttm[1], support_shape_ttm[2], support_shape_ttm[3])
        target_domain_features = target_domain_features.permute(0, 2, 1, 3, 4)
        target_domain_features = self.avgpool(target_domain_features).squeeze()
        return target_domain_features.permute(0, 2, 1)

    def get_feats(self, image_features):
        features = self.backbone(image_features)[-1]
        shape = features.shape
        features = features.reshape(int(shape[0] / self.argss.seq_len), self.argss.seq_len,  shape[1], shape[2], shape[3])
        features = features.permute(0, 2, 1, 3, 4)
        features = self.avgpool(features).squeeze().permute(0,2,1)
        return features

    def enhance_source(self, support_features, target_domain_features):
        b_s, seq_size, d_size = target_domain_features.shape
        # 第一维度是池子中的视频帧数量, 例如 25*8, 1024
        target_domain_features = target_domain_features.reshape(b_s * seq_size, d_size)
        b_s, seq_size, d_size = support_features.shape
        enhanced_support_features = support_features.clone()
        # support_features = [x for x in support_features]
        for i, support_feature in enumerate(support_features):
            # 8 1024    25*8 1024  =>  8*25,  8
            s2t_sim = cos_sim(support_feature, target_domain_features)
            max_indices = torch.argmax(s2t_sim)
            row = max_indices // s2t_sim.shape[1] - 1
            column = max_indices % s2t_sim.shape[1] - 1
            enhanced_support_features[i][row] = target_domain_features[column]
            # 给最大值出现的行全部置零

            s2t_sim[row] = 0

            max_indices = torch.argmax(s2t_sim)
            row = max_indices // s2t_sim.shape[1] - 1
            column = max_indices % s2t_sim.shape[1] - 1
            enhanced_support_features[i][row] = target_domain_features[column]
        return 0.2*enhanced_support_features + 0.8*support_features


    def student2inner_teacher(self, cum_dist_g, target_labels, support_features_comb_info, support_features_g,  support_labels, query_features_comb_info, query_features_g):
        a1 = self.aa[0]
        a2 = self.aa[1]
        # student网络的souce domain中 全局query序列和support原型序列的对比概率分布
        P_global = F.softmax(cum_dist_g, dim=-1)

        query_features_comb, q_bs, q_sub_seq_num, q_sub_seq_len = query_features_comb_info  # 25*5,4,2048
        support_features_comb, s_bs, s_sub_seq_num, s_sub_seq_len = support_features_comb_info  # 25*5,4,2048
        #给每个子序列都扩展一个文本向量
        #context_support= context_support.repeat_interleave(t_sub_seq_num, dim=0)
        # 2. student和inner_teacher之间
        ##########################################################################################################################################
        ######*****************inner_teacher和student之间的知识差，用self_loss 和 cross_loss来表示**************************########
        # 生成source domain中support和query的全序列的子序列

        # 同域特征teacher来进行特征提取 support和target子序列特征
        target_features_result = self.inner_teacher(query_features_comb, query_features_comb, query_features_comb)

        #support_features_comb = torch.cat([context_support, support_features_comb], dim=1)
        support_features_result = self.inner_teacher(support_features_comb, support_features_comb, support_features_comb)[:, :q_sub_seq_len, :]


        # 主网络的全局support原型和辅助网络的局部query比较
        # 全局的support原型维度  5 8 2048， 辅助网络的局部query 25*5, 4, 2048 (其中5是一共有多少个子序列， 4是一个子序列中的帧集合)
        local_dist_l2g = self.otam_distance(support_features_g, target_features_result)  # 125 5, 第一位是targt 第二维是support
        local_dist_l2g = local_dist_l2g.reshape(q_bs, q_sub_seq_num, -1)  # 25, 5, 5 #第一维度是target样本，第二维是每个target样本对应的子序列，第三位是 support类别
        P_local_l2g = F.softmax(-local_dist_l2g, dim=-1)  # 根据类别这个维度计算 局部query对比全局原型的 概率分布
        # support子序列原型计算
        unique_labels = torch.unique(support_labels)
        support_features_result = support_features_result.reshape(s_bs, s_sub_seq_num, s_sub_seq_len, -1)  # 15*5,4,2048 => 25, 5, 4, 2048
        support_features_result = [torch.mean(torch.index_select(support_features_result, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features_result = torch.stack(support_features_result)  # 获得每一个子序列的原型 5,5,4,2048， 第二个5是子序列的个数
        support_features_result = support_features_result.reshape(self.argss.way * s_sub_seq_num, s_sub_seq_len, -1)  # 25,4,2048
        # 辅助网络的局部support原型和主网络的全局query比较
        # 局部的support原型维度
        local_dist_g2l = self.otam_distance(support_features_result, query_features_g)  # 25  25  第一维度是25个target 第二维是5个support，每个support对应的5个子序列的原型
        '''
          target1    子序列1（类1 类2 类3 类4 类5） 子序列2（类1 类2 类3 类4 类5） 子序列3（类1 类2 类3 类4 类5） 子序列4（类1 类2 类3 类4 类5） 子序列5（类1 类2 类3 类4 类5）
          target2    子序列1（类1 类2 类3 类4 类5） 子序列2（类1 类2 类3 类4 类5） 子序列3（类1 类2 类3 类4 类5） 子序列4（类1 类2 类3 类4 类5） 子序列5（类1 类2 类3 类4 类5）
          ...
          target25   子序列1（类1 类2 类3 类4 类5） 子序列2（类1 类2 类3 类4 类5） 子序列3（类1 类2 类3 类4 类5） 子序列4（类1 类2 类3 类4 类5） 子序列5（类1 类2 类3 类4 类5）                  
        '''

        local_dist_g2l = local_dist_g2l.reshape(q_bs, q_sub_seq_num, -1)  # 25 5 5  第一维度是target样本，第二维是support子序列，第三位是每个support 类别
        # local特征取均值来计算损失  self的loss             5 8 1024  otam   100 2 1024
        P_local_g2l = F.softmax(-local_dist_g2l, dim=-1)  # 根据类别这个维度计算全局query对比局部原型的 概率分布 25 5 5

        # 同一个样本的全局向局部对准的类的分布  向该全局样本向全局样本对准分布概率 的KL散度
        self_loss_g2l = sum([-torch.sum(torch.mul(torch.log(P_local_g2l[index]), P_global[index])) for index in range(q_bs)])
        # 同一个样本的局部向全局对准的类的分布  向该全局样本向全局样本对准分布概率 的KL散度
        self_loss_l2g = sum([-torch.sum(torch.mul(torch.log(P_local_l2g[index]), P_global[index])) for index in range(q_bs)])
        # kl12=F.kl_div((-cum_dist1).softmax(-1).log(),(-cum_dist2).softmax(-1),reduction='sum')
        # kl21=F.kl_div((-cum_dist2).softmax(-1).log(),(-cum_dist1).softmax(-1),reduction='sum')
        self_loss =  a1*self_loss_l2g + a2*self_loss_g2l
        ######################################################################################################################

        ######################################################################################################################

        # cross_loss
        # 先获取target 下标，方便处理
        shuffle_P_local_g2l = []
        shuffle_P_local_l2g = []
        target_label_dict = {}
        for i, v in enumerate(target_labels):
            if target_label_dict.get(v.item()):
                target_label_dict[v.item()].append(i)
            else:
                target_label_dict[v.item()] = [i]
        for i, value in enumerate(target_labels):
            # 提取classlabel相同的，获取index，index不同继续抽
            while True:
                mid = random.choice(target_label_dict[value.item()])
                if int(i) != mid:
                    break
                else:
                    pass
            shuffle_P_local_g2l.append(P_local_g2l[(mid):((mid + 1))])
            shuffle_P_local_l2g.append(P_local_l2g[(mid):((mid + 1))])
        shuffle_P_local_g2l = torch.concat(shuffle_P_local_g2l, dim=0)
        shuffle_P_local_l2g = torch.concat(shuffle_P_local_l2g, dim=0)
        # 同类的其他样本的局部样本 向全局 该样本 分布概率 对齐的KL散度
        cross_loss_g2l = sum([-torch.sum(torch.mul(torch.log(shuffle_P_local_g2l[index]), P_global[index])) for index in range(q_bs)])
        # 同类的其他样本的局部样本 向全局 该样本 分布概率 对齐的KL散度
        cross_loss_l2g = sum([-torch.sum(torch.mul(torch.log(shuffle_P_local_l2g[index]), P_global[index])) for index in range(q_bs)])
        cross_loss = a1*cross_loss_l2g + a2*cross_loss_g2l

        return cross_loss/q_bs, self_loss/q_bs, local_dist_g2l, local_dist_l2g
    def sub_seq_generate(self, domain_features):
        combinations = self.Subsequence_Generator()
        domain_features_comb = torch.stack([torch.index_select(domain_features, 1, each) for each in combinations], dim=1)
        bs, sub_seq_num, sub_seq_len, _ = domain_features_comb.shape #25 10 4 512, 25是一个episode中的样本数， 10是子序列个数，4是子序列长度，

        domain_features_comb = domain_features_comb.reshape(bs * sub_seq_num, sub_seq_len, -1)  #250, 4, 512    第一维是以序列个数作为样本个数
        return [domain_features_comb, bs, sub_seq_num, sub_seq_len]

    def text_eh_temporal_transformer(self,  support_features, target_features, support_labels):

        #给support加上语义信息
        #support_features = torch.cat([context_support, support_features], dim=1)

        target_features = self.student_encoder(target_features, target_features, target_features)
        support_features = self.student_encoder(support_features, support_features, support_features)
        unique_labels = torch.unique(support_labels)
        support_features_pro = [
            torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        support_features_pro = torch.stack(support_features_pro)
        return support_features_pro, support_features, target_features

    def otam_distance(self, support_features, target_features):
        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]
        support_features = rearrange(support_features, 'b s d -> (b s) d')  # 5 8 1024-->40  1024
        target_features = rearrange(target_features, 'b s d -> (b s) d')
        frame_sim = cos_sim(target_features, support_features)  # 类别数量*每个类的样本数量， 类别数量
        frame_dists = 1 - frame_sim
        # dists维度为 query样本数量， support类别数量，帧数，帧数
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 25, 8, 8]
        # calculate query -> support and support -> query  双向匹配还是单向匹配
        if self.argss.SINGLE_DIRECT:
            cum_dists = OTAM_cum_dist_v2(dists)
        else:
            cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
        return cum_dists

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())

    def distribute_model(self):

        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        gpus_use_number = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if self.argss.num_gpus > 1:
            self.backbone.cuda()
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(gpus_use_number)])





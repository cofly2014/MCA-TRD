import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
import os
import torch
from torch.autograd import Variable


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


from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


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


class DSN_TEMPORAL(nn.Module):
    """
    OTAM with a CNN backbone.
    """

    def __init__(self, args):
        super(DSN_TEMPORAL, self).__init__()
        self.argss = args

        ##############################################################################

        #######################时序网络定义##############################################

        self.mid_dim = 576
        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)


        self.source_encoder = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.argss.TRANSFORMER_DEPTH))
        self.target_encoder = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.argss.TRANSFORMER_DEPTH))
        self.shared_encoder = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.argss.TRANSFORMER_DEPTH))
        self.shared_decoder = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.argss.TRANSFORMER_DEPTH))
        self.trans_target   = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.argss.TRANSFORMER_DEPTH))


        #########################end 时序网络定义##########################################

        ######################监督学习的线性分类器###########################################
        self.classification_layer = nn.Sequential(
            nn.LayerNorm(self.mid_dim),
            nn.Linear(self.mid_dim, self.argss.class_num)
        )
        ##################################################################################

        self.class_num = self.argss.class_num

        #shared_encoder中进行域判断的网络
        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('fc_se1', nn.Linear(in_features=self.mid_dim, out_features=100))
        self.shared_encoder_pred_domain.add_module('relu_se', nn.ReLU(True))
        # classify two domain
        self.shared_encoder_pred_domain.add_module('fc_se2', nn.Linear(in_features=100, out_features=2))

        self.specific_encoder_pred_domain = nn.Sequential()
        self.specific_encoder_pred_domain.add_module('fc_p_se1', nn.Linear(in_features=self.mid_dim, out_features=100))
        self.specific_encoder_pred_domain.add_module('relu_p_se', nn.ReLU(True))
        self.specific_encoder_pred_domain.add_module('fc_p_se2', nn.Linear(in_features=100, out_features=2))

        self.loss_similarity = torch.nn.CrossEntropyLoss()


        self.register_buffer('target_center_global', torch.zeros(self.argss.seq_len, self.mid_dim))

        self.momentum = self.argss.momentum


    def forward(self, source_domain_features, target_domain_features, support_num, support_labels, target_domain_centor_feature, iteration=1):  # 获得support support labels, query, support real class



        ######################################################################################
        #源域和目标域的domain-irrelevant的特征提取
        target_domain_shared_enc = self.shared_encoder(target_domain_features, target_domain_features, target_domain_features)
        source_domain_shared_enc = self.shared_encoder(source_domain_features, source_domain_features, source_domain_features)
        ##########################################################################################################################
        #源域和目标域的domain-specific的特征提取
        source_domain_private_enc = self.source_encoder(source_domain_features, source_domain_features, source_domain_features)
        target_domain_private_enc = self.target_encoder(target_domain_features, target_domain_features, target_domain_features)
        ######################################################################################
        #(1) 按照 source domain和target domain中 domain-irrelevant的特征来判断是哪个域 Domain classifier 按照源域和目标域中的shared的特征 进行域分类，目标是让其分不清其来自于哪个域，所以有一个梯度反转层
        #total_dann = self.label_pre_shared_domain_reverse(source_domain_shared_enc, target_domain_shared_enc)
        total_dann = self.label_pre_shared_domain_KL(source_domain_shared_enc, target_domain_shared_enc)
        ######################################################################################
        #(2)source domain中域不相关和域相关正交，target domain中域不相关和域相关正交 source domain的domain specific和domain irrelevent之间要正交, target domain的domain specific和domain irrelevent之间要正交，
        #total_diff = self.specific_domain_ortho_pre(source_domain_private_enc, source_domain_shared_enc, target_domain_private_enc, target_domain_shared_enc)
        total_diff =  self.label_pre_specific_domain(source_domain_private_enc, target_domain_private_enc)
        ######################################################################################
        #（3）循环一致性重构， 通过decoder用domain-irrelevant和domain-specific的特征来重构 encoder之前的原始特征
        total_domain_recon = self.domain_recons(source_domain_features, source_domain_private_enc, source_domain_shared_enc, target_domain_features, target_domain_private_enc, target_domain_shared_enc)
        #####################################################################################################################
        #target_domain_features_mean_s = torch.mean(target_domain_shared_enc, dim=0)
        #更新目标域中心
        #self.target_center_global.data  =  self.momentum * self.target_center_global + (1 - self.momentum) * target_domain_features_mean_s
        #target_domain_features_mean_s = self.target_center_global.data
        target_domain_centor_feature = self.shared_encoder(target_domain_centor_feature, target_domain_centor_feature, target_domain_centor_feature)
        target_domain_features_mean_s = target_domain_centor_feature.squeeze(0)

        source_num  = source_domain_features.shape[0]
        target_domain_features_mean_s = target_domain_features_mean_s.unsqueeze(0).expand(source_num, -1, -1)
        source_domain_shared_enc =  self.trans_target(target_domain_features_mean_s, source_domain_shared_enc, source_domain_shared_enc)
        #######################################################################################################################

        support_shared_enc, query_shared_enc = source_domain_shared_enc[0:support_num], source_domain_shared_enc[support_num:]
        unique_labels = torch.unique(support_labels)
        support_shared_enc_pro = [
            torch.mean(torch.index_select(support_shared_enc, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        support_shared_enc_pro = torch.stack(support_shared_enc_pro)

        cum_dist_g = -self.otam_distance(support_shared_enc_pro, query_shared_enc)  # 全局对比全局
        # 监督学习，对用backbone中提取的源域的数据进行分类
        class_logits = self.classification_layer(source_domain_shared_enc.mean(1))
        ##########################################################################################

        return_dict = {
            'class_logits': class_logits,
            'meta_logits': cum_dist_g,
            't_irre_pre_loss': total_dann,
            't_spec_pre_loss': total_diff,
            't_domain_recon_loss': total_domain_recon
        }  # [5， 5] , [10 64]

        return return_dict

    def forward_finetuning(self, source_domain_features, support_num , support_labels):  # 获得support support labels, query, support real class


        ######################################################################################
        #源域和目标域的domain-irrelevant的特征提取, 只需要微雕最后用于分类的特征的编码器参数即可
        source_domain_shared_enc = self.shared_encoder(source_domain_features, source_domain_features, source_domain_features)
        ##########################################################################################################################

        #####################################################################################################################
        support_shared_enc, query_shared_enc = source_domain_shared_enc[0:support_num], source_domain_shared_enc[support_num:]
        unique_labels = torch.unique(support_labels)
        support_shared_enc_pro = [
            torch.mean(torch.index_select(support_shared_enc, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        support_shared_enc_pro = torch.stack(support_shared_enc_pro)

        cum_dist_g = -self.otam_distance(support_shared_enc_pro, query_shared_enc)  # 全局对比全局
        # 监督学习，对用backbone中提取的源域的数据进行分类
        class_logits = self.classification_layer(source_domain_shared_enc.mean(1))
        ##########################################################################################

        return_dict = {
            'class_logits': class_logits,
            'meta_logits': cum_dist_g,
            'source_shared_code': source_domain_shared_enc
        }  # [5， 5] , [10 64]
        return return_dict

    def domain_recons(self, source_domain_features, source_domain_private_enc, source_domain_shared_enc, target_domain_features, target_domain_private_enc, target_domain_shared_enc):
        merged_source_feature_enc = source_domain_private_enc + source_domain_shared_enc
        merged_target_feature_enc = target_domain_private_enc + target_domain_shared_enc
        recon_source_feature = self.shared_decoder(merged_source_feature_enc, merged_source_feature_enc, merged_source_feature_enc)
        recon_target_feature = self.shared_decoder(merged_target_feature_enc, merged_target_feature_enc, merged_target_feature_enc)
        source_domain_mse = self.loss_recon_mse(recon_source_feature, source_domain_features)
        target_domain_mse = self.loss_recon_mse(recon_target_feature, target_domain_features)
        total_domain_recon = source_domain_mse + target_domain_mse
        #print("shared and specific domain Reconstruct loss: {}".format(total_domain_recon))
        return total_domain_recon

    ###############################################################################################################################################
    def label_pre_shared_domain_reverse(self, source_domain_shared_enc, target_domain_shared_enc):
        reversed_source_shared_code = ReverseLayerF.apply(source_domain_shared_enc, 0.2)
        source_domain_label = self.shared_encoder_pred_domain(torch.mean(reversed_source_shared_code, dim=1))
        reversed_target_shared_code = ReverseLayerF.apply(target_domain_shared_enc, 0.2)
        target_domain_label = self.shared_encoder_pred_domain(torch.mean(reversed_target_shared_code, dim=1))
        target_batch_size = source_domain_shared_enc.shape[0]
        domain_label = torch.zeros(target_batch_size)
        domain_label = domain_label.long().cuda()
        target_domainv_label = Variable(domain_label)

        source_batch_size = target_domain_shared_enc.shape[0]
        domain_label = torch.ones(source_batch_size)
        domain_label = domain_label.long().cuda()
        source_domainv_label = Variable(domain_label)

        # 是loss_similarity是cross entropy loss
        target_dann = self.loss_similarity(target_domain_label, target_domainv_label)
        source_dann = self.loss_similarity(source_domain_label, source_domainv_label)
        total_dann = target_dann + source_dann
        #print("shared Domain classify loss: {}".format(total_dann))
        return total_dann

    '''
    让通过域不相关的特征预测出的域分布逼近 [0.5,0.5] 换句话说就是让域不可分。
    '''
    def label_pre_shared_domain_KL(self, source_domain_shared_enc,  target_domain_shared_enc):

        source_domain_label = self.shared_encoder_pred_domain(torch.mean(source_domain_shared_enc, dim=1))
        target_domain_label = self.shared_encoder_pred_domain(torch.mean(target_domain_shared_enc, dim=1))

        target_batch_size = target_domain_shared_enc.shape[0]
        shared_target_domain_label = torch.zeros(target_batch_size,2) + 0.5
        shared_target_domain_label = shared_target_domain_label.cuda()
        shared_target_domainv_label = Variable(shared_target_domain_label)

        source_batch_size = source_domain_shared_enc.shape[0]
        shared_source_domain_label = torch.ones(source_batch_size,2) - 0.5
        shared_source_domain_label = shared_source_domain_label.cuda()
        shared_source_domainv_label = Variable(shared_source_domain_label)

        target_dann = F.kl_div(F.log_softmax(target_domain_label, dim=1),  shared_target_domainv_label, reduction='mean')
        source_dann = F.kl_div(F.log_softmax(source_domain_label, dim=1),  shared_source_domainv_label, reduction='mean')

        temporal_dann_loss  = target_dann + source_dann
        #print("shared Domain classify loss: {}".format(temporal_dann_loss))
        return temporal_dann_loss

    ###############################################################################################################################################

    ###############################################################################################################################################
    def specific_domain_ortho_pre(self, source_domain_private_enc, source_domain_shared_enc, target_domain_private_enc, target_domain_shared_enc):
        source_diff = self.loss_diff_temporal(source_domain_private_enc, source_domain_shared_enc)
        target_diff = self.loss_diff_temporal(target_domain_private_enc, target_domain_shared_enc)
        total_diff = source_diff + target_diff
        print("specific Domain orthogonal diff: {}".format(total_diff))
        return total_diff

    def label_pre_specific_domain(self, source_domain_private_enc, target_domain_private_enc):
        source_domain_label = self.specific_encoder_pred_domain(torch.mean(source_domain_private_enc, dim=1))
        target_domain_label = self.specific_encoder_pred_domain(torch.mean(target_domain_private_enc, dim=1))

        target_batch_size = target_domain_private_enc.shape[0]
        domain_label = torch.zeros(target_batch_size)
        domain_label = domain_label.long().cuda()
        target_domainv_label = Variable(domain_label)

        source_batch_size = source_domain_private_enc.shape[0]
        domain_label = torch.ones(source_batch_size)
        domain_label = domain_label.long().cuda()
        source_domainv_label = Variable(domain_label)

        # 是loss_similarity是cross entropy loss
        target_diff_loss = self.loss_similarity(target_domain_label, target_domainv_label)
        source_diff_loss = self.loss_similarity(source_domain_label, source_domainv_label)

        '''
        target_ = torch.stack( ( torch.zeros(target_batch_size), torch.ones(target_batch_size) ), dim=1).cuda()
        source_ = torch.stack( ( torch.ones(source_batch_size),  torch.zeros(source_batch_size)), dim=1).cuda()
        target_dann = F.kl_div(F.log_softmax(target_domain_label),  target_, reduction='mean')
        source_dann = F.kl_div(F.log_softmax(source_domain_label),  source_, reduction='mean')
        '''

        temporal_diff_loss = target_diff_loss + source_diff_loss
        #print("specific Domain classify loss: {}".format(temporal_diff_loss))
        return temporal_diff_loss
    ###############################################################################################################################################
    def forward_test_bk(self, source_domain_features, target_domain_features, support_num, support_labels, iteration=1):  # 获得support support labels, query, support real class

        ######################################################################################
        #源域和目标域的domain-irrelevant的特征提取
        target_domain_shared_enc = self.shared_encoder(target_domain_features, target_domain_features, target_domain_features)
        source_domain_shared_enc = self.shared_encoder(source_domain_features, source_domain_features, source_domain_features)
        ##########################################################################################################################

        #####################################################################################################################
        target_domain_features_mean_s = torch.mean(target_domain_shared_enc, dim=0)
        #更新目标域中心
        self.target_center_global.data  =  self.momentum * self.target_center_global + (1 - self.momentum) * target_domain_features_mean_s
        target_domain_features_mean_s = self.target_center_global.data

        source_num  = source_domain_features.shape[0]
        target_domain_features_mean_s = target_domain_features_mean_s.unsqueeze(0).expand(source_num, -1, -1)
        source_domain_shared_enc =  self.trans_target(target_domain_features_mean_s, source_domain_shared_enc, source_domain_shared_enc) + source_domain_shared_enc
        #######################################################################################################################

        support_shared_enc, query_shared_enc = source_domain_shared_enc[0:support_num], source_domain_shared_enc[support_num:]
        unique_labels = torch.unique(support_labels)
        support_shared_enc_pro = [
            torch.mean(torch.index_select(support_shared_enc, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        support_shared_enc_pro = torch.stack(support_shared_enc_pro)

        cum_dist_g = -self.otam_distance(support_shared_enc_pro, query_shared_enc)  # 全局对比全局
        # 监督学习，对用backbone中提取的源域的数据进行分类
        class_logits = self.classification_layer(source_domain_shared_enc.mean(1))
        ##########################################################################################

        return_dict = {
            'class_logits': class_logits,
            'meta_logits': cum_dist_g
        }  # [5， 5] , [10 64]

        return return_dict

    def forward_test(self, source_domain_features, support_num , support_labels):  # 获得support support labels, query, support real class


        ######################################################################################
        #源域和目标域的domain-irrelevant的特征提取
        source_domain_shared_enc = self.shared_encoder(source_domain_features, source_domain_features, source_domain_features)
        ##########################################################################################################################
        #源域和目标域的domain-specific的特征提取
        source_domain_private_enc = self.source_encoder(source_domain_features, source_domain_features, source_domain_features)
        ######################################################################################

        #####################################################################################################################
        support_shared_enc, query_shared_enc = source_domain_shared_enc[0:support_num], source_domain_shared_enc[support_num:]
        unique_labels = torch.unique(support_labels)
        support_shared_enc_pro = [
            torch.mean(torch.index_select(support_shared_enc, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        support_shared_enc_pro = torch.stack(support_shared_enc_pro)

        cum_dist_g = -self.otam_distance(support_shared_enc_pro, query_shared_enc)  # 全局对比全局
        # 监督学习，对用backbone中提取的源域的数据进行分类
        class_logits = self.classification_layer(source_domain_shared_enc.mean(1))
        ##########################################################################################

        return_dict = {
            'class_logits': class_logits,
            'meta_logits': cum_dist_g
        }  # [5， 5] , [10 64]

        return return_dict

    '''
    def get_feats(self, image_features):
        features = self.backbone(image_features)[-1]
        shape = features.shape
        features = features.reshape(int(shape[0] / self.argss.seq_len), self.argss.seq_len,  shape[1], shape[2], shape[3])
        features = features.permute(0, 2, 1, 3, 4)
        features = self.avgpool(features).squeeze().permute(0,2,1)
        return features
    '''

    def text_eh_temporal_transformer(self,  support_features, target_features, support_labels):

        #给support加上语义信息
        #support_features = torch.cat([context_support, support_features], dim=1)

        target_features = self.shared_encoder(target_features, target_features, target_features)
        support_features = self.shared_encoder(support_features, support_features, support_features)
        unique_labels = torch.unique(support_labels)
        support_features_pro = [
            torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
            for c in unique_labels]
        support_features_pro = torch.stack(support_features_pro)
        return support_features_pro, support_features, target_features

    def loss_diff_temporal(self, input1, input2):
        input1 = torch.mean(input1, dim=2)
        input2 = torch.mean(input2, dim=2)

        batch_size = input1.size(0)
        frames_size = input1.size(1)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss

    def loss_recon_mse(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

    def loss_recon_simse(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)
        return simse

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

        self.shared_encoder.cuda()
        self.shared_encoder= torch.nn.DataParallel(self.shared_encoder, device_ids=[i for i in range(gpus_use_number)])

        self.shared_decoder.cuda()
        self.shared_decoder= torch.nn.DataParallel(self.shared_decoder, device_ids=[i for i in range(gpus_use_number)])

        self.source_encoder.cuda()
        self.source_encoder= torch.nn.DataParallel(self.source_encoder, device_ids=[i for i in range(gpus_use_number)])

        self.target_encoder.cuda()
        self.target_encoder= torch.nn.DataParallel(self.target_encoder, device_ids=[i for i in range(gpus_use_number)])

        self.trans_target.cuda()
        self.trans_target= torch.nn.DataParallel(self.trans_target, device_ids=[i for i in range(gpus_use_number)])




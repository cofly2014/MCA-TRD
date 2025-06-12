import torch
import os
import torch.nn as nn
from models.base.d830.temporal_model_compat_gg import DSN_TEMPORAL
from models.base.resNet import MyResNet
import numpy as np
class DSN_CROSS_FSAR(nn.Module):
    """
    OTAM with a CNN backbone.
    """

    def __init__(self, args):
        super(DSN_CROSS_FSAR, self).__init__()
        self.argss = args

        self.argss.num_patches = 16
        self.argss.reduction_fac = 4

        self.mid_dim = 576
        #骨干网络，用来提取每一帧的特征
        #naive监督分类器，这个是否需要，需要验证
        self.fc_norm = nn.LayerNorm(self.mid_dim)
        self.class_num = self.argss.class_num
        #根据我们的任务修改之后的DSN网络,其定义在model_compat中
        self.temporal_dsn_net = DSN_TEMPORAL(self.argss)

        self.backbone = MyResNet(self.argss)
        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, inputs, iteration=1):  # 获得support support labels, query, support real class

        support_images, support_labels, target_images, support_real_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels']
        target_domain_centor_g = inputs['target_domain_centor_g']
        target_labels = inputs["target_labels"]
        target_domain_set = inputs['target_domain_set']  # 在训练阶段这个有用
        source_domain_set = torch.concat([support_images, target_images], dim=0)
        support_num = int(support_images.shape[0] / self.argss.seq_len)


        support_features = self.get_feats_t(support_images)
        query_features = self.get_feats_t(target_images)
        target_domain_features = self.get_feats_t(target_domain_set)
        source_domain_features = torch.concat([support_features, query_features], dim=0)

        target_domain_centor_feature = self.get_feats_t(target_domain_centor_g)

        t_result = self.temporal_dsn_net(source_domain_features, target_domain_features, support_num , support_labels, target_domain_centor_feature)

        return t_result


    def forward_test(self, inputs, iteration=1):  # 获得support support labels, query, support real class

        support_images, support_labels, target_images, support_real_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels']
        target_labels = inputs["target_labels"]
        target_domain_set = inputs['target_domain_set']  # 在训练阶段这个有用
        source_domain_set = torch.concat([support_images, target_images], dim=0)
        support_num = int(support_images.shape[0] / self.argss.seq_len)

        support_features = self.get_feats_t(support_images)
        query_features = self.get_feats_t(target_images)
        target_domain_features = self.get_feats_t(target_domain_set)

        source_domain_features = torch.concat([support_features, query_features], dim=0)
        t_result = self.temporal_dsn_net.forward_test(source_domain_features, support_num , support_labels)
       # t_result = self.temporal_dsn_net.forward_test(source_domain_features, target_domain_features, support_num, support_labels)

        return t_result

    def forward_finetuning(self, inputs, iteration=1):  # 获得support support labels, query, support real class

        support_images, support_labels, target_images, support_real_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels']
        target_labels = inputs["target_labels"]
        target_domain_set = inputs['target_domain_set']  # 在训练阶段这个有用
        source_domain_set = torch.concat([support_images, target_images], dim=0)
        support_num = int(support_images.shape[0] / self.argss.seq_len)

        support_features = self.get_feats_t(support_images)
        query_features = self.get_feats_t(target_images)
        source_domain_features = torch.concat([support_features, query_features], dim=0)
        t_s_result = self.temporal_dsn_net.forward_finetuning(source_domain_features, support_num ,support_labels)

        return t_s_result


    def get_feats(self, image_features):
        features = self.backbone(image_features)[-1]


        features = self.adap_max(features)
        features = features.reshape(-1, self.argss.trans_linear_in_dim, self.argss.num_patches)

        features = features.permute(0, 2, 1)

        return features

    def get_feats_t_bk(self, image_features):
        features = self.backbone(image_features)[-1]
        shape = features.shape
        features = features.reshape(int(shape[0] / self.argss.seq_len), self.argss.seq_len,  shape[1], shape[2], shape[3])
        features = self.avgpool(features).squeeze()
        return features

    def get_feats_t(self, image_features):
        features_array = self.backbone(image_features)
        features_ = features_array[-4]
        features = features_array[-1]

        features = self.avgpool(features).squeeze()
        features_ = self.avgpool(features_).squeeze()

        features  = torch.cat([features, features_], dim = -1)
        shape = features.shape
        features = features.reshape(int(shape[0] / self.argss.seq_len), self.argss.seq_len,  shape[1])

        return features

    def distribute_model(self):

        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        gpus_use_number = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        self.backbone.cuda()
        self.backbone= torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(gpus_use_number)])
        self.temporal_dsn_net.distribute_model()



import torch
import torch.nn.functional as F


def compute_similarity(a, b):
    # a: (batch_size, video_seq_len, dim)
    # b: (batch_size, subseq_seq_len, dim)
    # 计算余弦相似度
    a = F.normalize(a, p=2, dim=-1)  # (batch_size, video_seq_len, dim)
    b = F.normalize(b, p=2, dim=-1)  # (batch_size, subseq_seq_len, dim)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(a, b.transpose(-1, -2))  # (batch_size, video_seq_len, subseq_seq_len)

    return similarity_matrix


def compute_video_to_subseq_similarity(video_features, subseq_features):
    # video_features: (batch_size, 8, 512)
    # subseq_features: (batch_size, 4, 512)
    batch_size, video_seq_len, dim = video_features.size()
    _, subseq_seq_len, _ = subseq_features.size()

    # 计算相似度矩阵
    similarity_matrix = compute_similarity(video_features,
                                           subseq_features)  # (batch_size, video_seq_len, subseq_seq_len)

    # 对于每个视频帧，获取与子序列帧的最大相似度
    max_sim = similarity_matrix.max(dim=2)[0].mean(dim=1)  # (batch_size, video_seq_len) -> (batch_size)

    # 对于每个子序列帧，获取与视频帧的最大相似度
    max_sim_subseq = similarity_matrix.max(dim=1)[0].mean(dim=1)  # (batch_size, subseq_seq_len) -> (batch_size)

    # 计算最终相似度度量
    final_similarity = (max_sim + max_sim_subseq) / 2
    return final_similarity


# 示例数据
batch_size = 1
video_seq_len = 8
subseq_seq_len = 4
dim = 512

# 生成示例数据
video_features = torch.randn(batch_size, video_seq_len, dim)
subseq_features = torch.randn(batch_size, subseq_seq_len, dim)

# 计算相似度
similarity = compute_video_to_subseq_similarity(video_features, subseq_features)
print(f'Similarity: {similarity}')

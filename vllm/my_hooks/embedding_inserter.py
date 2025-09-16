# my_hooks/embedding_inserter.py
from typing import Any, List
import torch
import os
import datetime

# 日志文件路径，存放在当前脚本所在目录下
LOG_FILE = os.path.join(os.path.dirname(__file__), "embedding_inserter.log")

def log_event(message: str):
    """
    将信息写入日志文件，带时间戳。

    Args:
        message (str): 需要写入的日志信息
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] {message}\n")


def process_and_insert_patches(outputs: List[Any],
                               seq_group_metadata_list: list,
                               seq_id_to_seq_group: dict,
                               enabled: bool = True,
                               image_token_id: int = 151655,
                               signal_token: str = "\n",
                               top_k: int = 32,
                               sim_threshold: float = 0.2):
    """
    主函数：在生成过程中，检测到特殊条件后，将与当前上下文相关的视觉 patch embedding 插入序列 embedding。

    Args:
        outputs (List[Any]): 模型原始输出
        seq_group_metadata_list (list): 包含每个序列组的元信息（如采样参数、tokenizer 等）
        seq_id_to_seq_group (dict): 序列 id 到序列组的映射（这里未直接使用）
        enabled (bool): 是否启用该 hook
        image_token_id (int): 图像占位 token 的 ID（例如 <image>）
        signal_token (str): 插入触发信号（例如 "\n" 表示换行符）
        top_k (int): 最多选择多少个相似 patch
        sim_threshold (float): 相似度阈值，超过才会被选中

    Returns:
        List[Any]: 修改后的 outputs
    """
    if not enabled:
        # 如果没启用，就直接返回原始输出
        log_event("Hook disabled, skipping patch insertion")
        return outputs

    # 遍历每个序列组（一个 batch 里可能有多个独立请求）
    for seq_group in seq_group_metadata_list:
        for seq in seq_group.seqs:  # 每个序列组可能有多个序列（例如 beam search 分支）
            data = seq.data
            token_ids = data.get_token_ids()       # 当前序列的 token id 列表
            embeds = data.get_token_embeddings()   # 对应的 token embedding
            if embeds is None or len(token_ids) == 0:
                continue  # 如果没有 embedding 或 token，就跳过

            # 找到所有 <image> token 出现的位置
            token_ids_tensor = torch.tensor(token_ids, device=embeds.device)
            image_indices = (token_ids_tensor == image_token_id).nonzero(as_tuple=True)[0]
            if image_indices.numel() == 0:
                continue  # 如果没有图像 token，就跳过

            # 检查最后一个 token 是否包含 signal_token（例如 "\n"）
            tokenizer = seq_group.sampling_params.tokenizer if seq_group.sampling_params else None
            if tokenizer:
                last_token_str = tokenizer.decode([token_ids[-1]], skip_special_tokens=False)
                if signal_token not in last_token_str:
                    continue  # 只有检测到 signal_token 才触发 patch 插入

            # 取出所有图像 token 对应的 embedding
            image_embeds = embeds[image_indices]

            # 取最后一个 token 的 hidden state
            last_hidden = embeds[-1].unsqueeze(0)  # [1, D]
            h = _l2norm(last_hidden)               # 归一化
            p = _l2norm(image_embeds)              # 归一化后的图像 patch 向量

            # 计算最后 token 和所有 image patch 的相似度
            sim_scores = torch.matmul(h, p.T).squeeze(0)  # [num_patches]

            # 根据阈值和 top-k 规则选择 patch
            patch_ids = _select_patches(sim_scores, top_k, sim_threshold)

            # 从 image_embeds 里取出被选中的 patch，并拼接到原始序列 embedding 后
            selected_patches = image_embeds.index_select(0, patch_ids).unsqueeze(0)
            new_embeds = torch.cat([embeds, selected_patches], dim=0)

            # 覆盖缓存的 embedding，使后续生成用到新拼接的 embedding
            data._cached_all_token_embeds = new_embeds

            # 写日志，记录插入了多少 patch
            log_event(f"Request {seq_group.request_id}: "
                      f"selected {len(patch_ids)} patches at step, "
                      f"indices={patch_ids.tolist()}")

    return outputs


def _l2norm(x, eps=1e-8):
    """
    对向量做 L2 归一化，避免除零。

    Args:
        x (Tensor): 输入向量
        eps (float): 防止除零的小常数

    Returns:
        Tensor: 单位化后的向量
    """
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _select_patches(sim_scores, top_k, threshold):
    """
    根据相似度选择 patch 索引。

    规则：
    1. 如果有分数超过阈值，则选出这些 patch
    2. 如果没有超过阈值，则选 top-k 个最高的
    3. 如果超过阈值的数量大于 top-k，则再取 top-k

    Args:
        sim_scores (Tensor): 相似度分数 [num_patches]
        top_k (int): 最多选择的数量
        threshold (float): 相似度阈值

    Returns:
        Tensor: 选中的 patch 索引
    """
    # 找出分数 >= 阈值的 patch
    patch_ids = (sim_scores >= threshold).nonzero(as_tuple=False).squeeze(1)

    if patch_ids.numel() == 0:
        # 如果没有 patch 过阈值，就取 top-k
        k = min(top_k, sim_scores.numel())
        _, patch_ids = sim_scores.topk(k=k, largest=True, sorted=False)
    elif patch_ids.numel() > top_k:
        # 如果超过 top-k，就在过阈值的里再取 top-k
        _, top_idx = sim_scores[patch_ids].topk(k=top_k, largest=True, sorted=False)
        patch_ids = patch_ids[top_idx]

    return patch_ids

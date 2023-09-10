import torch
from torch import nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """Compute loss for model. 

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """

        batch_size = features.shape[0]
   
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(features.device)    # [B, B]

        contrast_count = features.shape[1]  # 2
        # [B, 2, F] -> 2 * [B, F] -> [2B, F]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature   # [2B, F]
        anchor_count = contrast_count

        # compute logits
        # ([2B, F] * [F, 2B]) / temperature = [2B, 2B]
        # (Zi * Za) / tao, a=I
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # [2B, 1]
        logits = anchor_dot_contrast - logits_max.detach()  # [2B, 2B]

        # tile mask
        # mask: [B, B] repeat [2, 2] = [[B, B], [B, B]
        #                               [B, B], [B, B]]
        mask = mask.repeat(anchor_count, contrast_count)  # [2B, 2B]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(    # [2B, 2B]
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask   # [2B, 2B]

        # compute log_prob
        # exp(Zi * Za / tao) a={I/{i}}
        exp_logits = torch.exp(logits) * logits_mask    # [2B, 2B]
        # log{[exp(Zi*Za/tao)] / [sum(exp(Zi*Za/tao))]}
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  #[2B, 2B]

        # compute mean of log-likelihood over positive
        # sum(log_prob) / sum(P(i))
        loss = -(mask * log_prob).sum(1) / mask.sum(1)  # [2B,]
        loss = loss.view(anchor_count, batch_size).mean()   # [1,]

        return loss

        
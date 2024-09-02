import torch
import logging

class FusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(self, lm_head, pad_id=0, chunk_size=65536):
        """ Compute CrossEntropy loss for multi-stream LM using liger fused triton kernel """
        super(FusedLinearCrossEntropyLoss, self).__init__()
        self.lm_head = lm_head
        self.pad_id = pad_id
        self.chunk_size = chunk_size

        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
            self.loss = LigerFusedLinearCrossEntropyLoss()
            self.fused = True
        except:
            self.loss = torch.nn.CrossEntropyLoss(reduction='none')
            self.fused = False
            logging.warning("liger_kernel is not available. Use Pytorch implementation")

        self.torch_loss = torch.nn.CrossEntropyLoss(reduction='none')
        
    def __call__(self, hidden, targets):
        """
        hidden (torch.Tensor): hidden embeddings, typically output from transformer
          with lm_head bias. Size: (B, T, nq, D)
        targets (torch.Tensor): predicting target. Size (B, T, nq)
        """

        assert targets.size() == hidden.size()[:3]
        B, T, nq = targets.size()

        fused = self.fused and self.training

        # select items that are not padding.
        padding_mask = targets != self.pad_id
        hidden = hidden[padding_mask]
        targets = targets[padding_mask]

        # compute loss
        if fused: 
            logits = None
            loss = self.loss(self.lm_head.weight, hidden, targets)
        else:
            # chunk-by-chunk CE loss to avoid memory peak
            chunk_id, logits, loss = 0, [], []
            while chunk_id * self.chunk_size < len(hidden):
                start = chunk_id * self.chunk_size
                end = min((chunk_id + 1) * self.chunk_size, len(hidden))
                this_logits = self.lm_head(hidden[start: end])
                this_targets = targets[start: end]
                this_loss = self.torch_loss(this_logits, this_targets)
                logits.append(this_logits)
                loss.append(this_loss)
                chunk_id += 1
            loss = torch.cat(loss).mean()
        weight = targets.numel()
        stats = {"loss": loss.clone().detach(), "weight": weight}
        
        # compute token accuracy
        if not fused:
            logits = torch.cat(logits, dim=0)
            layer_idx = torch.arange(nq, device=hidden.device).tile(B, T, 1)
            layer_idx = layer_idx[padding_mask]

            for idx in range(nq):
                acc = torch.logical_and(
                    logits.argmax(-1) == targets,
                    layer_idx == idx
                ).float().sum()
                acc = acc / (layer_idx == idx).float().sum()
                stats[f"acc_layer{idx}"] = acc.clone().detach()

        return loss, logits, stats, weight

if __name__ == "__main__":
    hidden = torch.randn((1, 15, 2, 512)).float().cuda() * 100
    target = torch.randint(0, 70031, (1, 15, 2)).long().cuda()
    linear = torch.nn.Linear(512, 70032).cuda()

    liger_loss = FusedLinearCrossEntropyLoss(linear, pad_id=80000).cuda()
    torch_loss = torch.nn.CrossEntropyLoss(ignore_index=80000)

    loss_liger, _, _, _ = liger_loss(hidden, target)
    loss_torch = torch_loss(linear(hidden).view(-1, 70032), target.view(-1))

    print('loss_liger', 'loss_torch', loss_liger, loss_torch)




import torch
from torch import unique
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class Coverage(Metric):
    def __init__(self, codebook_size, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.codebook_size = codebook_size
        self.add_state("preds", default=[], dist_reduce_fx="cat")

    def update(self, codes):
        self.preds.append(codes)

    def compute(self):
        all_codes = dim_zero_cat(self.preds)
        unique_codes = unique(all_codes)
        return unique_codes.size(0) / self.codebook_size
    
class Collision(Metric):
    def __init__(self, items, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.items = items
        self.add_state("preds", default=[], dist_reduce_fx="cat")

    def update(self, codes):
        self.preds.append(codes)

    def compute(self):
        all_codes = dim_zero_cat(self.preds)
        unique_codes = unique(all_codes, dim=0)
        return (self.items-unique_codes.shape[0]) / self.items
    

class NDCG(Metric):
    def __init__(self, k: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k

        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, pos: torch.Tensor):
        self.scores.append(self._ndcg(pos))

    def compute(self):
        all_scores = dim_zero_cat(self.scores)

        return all_scores.mean()

    def _ndcg(self, pos_index: torch.Tensor):
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
        return dcg[:, :self.k].sum(dim=1)

        
class Recall(Metric):
    def __init__(self, k: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, pos: torch.Tensor):
        self.scores.append(self._recall(pos))

    def compute(self):
        all_scores = dim_zero_cat(self.scores)

        return all_scores.mean()

    def _recall(self, pos: torch.Tensor):
        return pos[:, :self.k].sum(dim=1).float()

        
class MRR(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_score", default=[], dist_reduce_fx="cat")

    def update(self, y_score: torch.Tensor, y_true: torch.Tensor):
        self.y_score.append(self._mrr_score(y_score, y_true))

    def compute(self):
        all_scores = dim_zero_cat(self.y_score)

        return all_scores.mean()

    def _mrr_score(self, y_score: torch.Tensor, y_true: torch.Tensor) -> float:
        if y_score.dim() == 1:
            return self._mrr_score(y_score.unsqueeze(0), y_true.unsqueeze(0))
        order = torch.argsort(y_score, descending=True)
        y_true_sorted = torch.gather(y_true, 1, order)
        rr_score = y_true_sorted / (torch.arange(y_true_sorted.shape[-1], dtype=y_true.dtype, device=y_true.device) + 1)
        mrr_ = rr_score.sum(dim=-1) / y_true.sum(dim=-1)
        
        return mrr_



    

    

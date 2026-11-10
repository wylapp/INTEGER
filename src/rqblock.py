import torch
import os
import json
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import torch.optim as optim
from src.metrics import *
from src.qmodel import ResidualVectorQuantization


class RQ_VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.r2vq = ResidualVectorQuantization(
            codebook_dim=self.config.codebook_dim,
            n_layers=len(self.config.codebook_size),
            codebook_size=self.config.codebook_size[0],
            beta=self.config.beta,
        )
    def forward(self, x, current_layer=None):
        codes_t, codes_v, quant, loss = self.r2vq(x, current_layer)
        return loss, quant, codes_t, codes_v
    
    def encode(self, x):
        return self.r2vq.encode(x)
    
class RQ_VAEPretrainer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = RQ_VAE(self.config)
        self.coverage1_t = Coverage(self.config.codebook_size[0])
        self.coverage2_t = Coverage(self.config.codebook_size[1])
        self.coverage3_t = Coverage(self.config.codebook_size[2])
        self.coverage4_t = Coverage(self.config.codebook_size[3])
        self.coverage1_v = Coverage(self.config.codebook_size[0])
        self.coverage2_v = Coverage(self.config.codebook_size[1])
        self.coverage3_v = Coverage(self.config.codebook_size[2])
        self.coverage4_v = Coverage(self.config.codebook_size[3])
        self.collision1 = Collision(self.config.item_count)
        self.collisiont = Collision(self.config.item_count)
        self.collisionv = Collision(self.config.item_count)
        self.step_out = []
        self.n_layers = len(self.config.codebook_size)
    
    def _current_layer(self) -> int:
        if not hasattr(self.config, 'layer_wise') or not self.config.layer_wise:
            return None 
        if self.global_step < 500:
            layer_idx = 0
        elif self.global_step < 3500:
            layer_idx = 1
        elif self.global_step < 8500:
            layer_idx = 2
        else:
            layer_idx = 3
        return layer_idx
    
    def forward(self, inputs, current_layer=None):
        return self.model(inputs, current_layer)

    def training_step(self, batch, batch_idx):

        current_layer = self._current_layer()
        loss, quant, codes_t, codes_v = self(batch, current_layer=current_layer)

        if current_layer is not None:
            self.log(f"r2vq/L_quant_layer{current_layer}", loss, 
                     on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        else:
            self.log("r2vq/L_quant", loss, on_step=True, on_epoch=True, 
                     prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        codes_t, codes_v = self.model.encode(batch)
        self.coverage1_t.update(codes_t[:, 0])
        self.coverage2_t.update(codes_t[:, 1])
        self.coverage3_t.update(codes_t[:, 2])
        self.coverage4_t.update(codes_t[:, 3])
        self.coverage1_v.update(codes_v[:, 0])
        self.coverage2_v.update(codes_v[:, 1])
        self.coverage3_v.update(codes_v[:, 2])
        self.coverage4_v.update(codes_v[:, 3])
        self.collisiont.update(codes_t)
        self.collisionv.update(codes_v)
        self.collision1.update(torch.concat([codes_t, codes_v], dim=-1))
        
    def on_validation_epoch_end(self):
        self.log_dict({'val_r2vq/usage1_t': self.coverage1_t.compute(),
                       'val_r2vq/usage2_t': self.coverage2_t.compute(),
                       'val_r2vq/usage3_t': self.coverage3_t.compute(),
                       'val_r2vq/usage4_t': self.coverage4_t.compute(),
                       'val_r2vq/usage1_v': self.coverage1_v.compute(),
                       'val_r2vq/usage2_v': self.coverage2_v.compute(),
                       'val_r2vq/usage3_v': self.coverage3_v.compute(),
                       'val_r2vq/usage4_v': self.coverage4_v.compute(),
                       'val_r2vq/collisiont': self.collisiont.compute(),
                       'val_r2vq/collisionv': self.collisionv.compute(),
                       'val_r2vq/collision': self.collision1.compute(),}, on_epoch=True, sync_dist=True)
        self.coverage1_t.reset()
        self.coverage2_t.reset()
        self.coverage3_t.reset()
        self.coverage4_t.reset()
        self.coverage1_v.reset()
        self.coverage2_v.reset()
        self.coverage3_v.reset()
        self.coverage4_v.reset()
        self.collision1.reset()
        self.collisiont.reset()
        self.collisionv.reset()

    def predict_step(self, batch, batch_idx):
        codes_t, codes_v = self.model.encode(batch)
        codes = torch.concat([codes_t, codes_v], dim=-1)
        self.step_out.append(codes)
        return codes
    
    def on_predict_epoch_end(self):
        self.step_out = torch.cat(self.step_out, dim=0)
        torch.save(self.step_out, os.path.join(self.config.data_dir, self.config.dataset, f"semantic_{self.config.sid_group_t}{self.config.sid_group_v}.pt"))
        cbs = [f'cb{j}_{i}' for j in range(len(self.config.codebook_size)*2) for i in range(256) ] + ['pad', 'eos']
        vocab = {c:i for i, c in enumerate(cbs)}
        with open(os.path.join(self.config.data_dir, self.config.dataset, f"vocabulary_{self.config.sid_group_t}{self.config.sid_group_v}.json"), "w") as f:
            json.dump(vocab, f, ensure_ascii=False)
    
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.rq_lr, weight_decay=self.config.rq_weight_decay)

        return {
            "optimizer": optimizer,
        }

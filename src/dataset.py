# import torch
import json
import torch
import lightning as L
import numpy as np
from pathlib import Path
from os import path
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from itertools import chain
from src.preprocessing import preprocess


class absSBRDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.config = args
        self.umap = json.load(open(path.join(self.config.data_dir, self.config.dataset, "umap.json")))
        self.smap = json.load(open(path.join(self.config.data_dir, self.config.dataset, "smap.json")))
        self.vocab = json.load(open(path.join(self.config.data_dir, self.config.dataset, f"vocabulary_{args.sid_group}.json")))
        self.semIDs = torch.load(
            path.join(
                self.config.data_dir,
                self.config.dataset,
                f"semantic_{args.sid_group}.pt",
            ),
            map_location="cpu",
        )
        if self.semIDs.ndim != 2 or self.semIDs.shape[1] % 2:
            raise ValueError(
                "Multimodal semantic IDs must have shape [num_items, 2 * num_codebooks]"
            )
        self.modality_sid_length = self.semIDs.shape[1] // 2
        
        self._preprocess()

    def _preprocess(self):
        self.text_sidmap = {}
        self.image_sidmap = {}
        for i, sid in tqdm(enumerate(self.semIDs), desc='generating semantic ID word pieces'):
            text_sid, image_sid = sid[:self.modality_sid_length], sid[self.modality_sid_length:]
            self.text_sidmap[i] = [
                self.vocab[f'cb{k}_{w}'] for k, w in enumerate(text_sid)
            ]
            self.image_sidmap[i] = [
                self.vocab[f'cb{k + self.modality_sid_length}_{w}']
                for k, w in enumerate(image_sid)
            ]

    def item_sid_pair(self, item):
        return self.text_sidmap[item], self.image_sidmap[item]

    def collate_history(self, histories, modality_index):
        input_ids = torch.full(
            (
                len(histories),
                self.config.max_history_length * self.modality_sid_length + 1,
            ),
            self.pad_id(),
            dtype=torch.long,
        )
        for row, history in enumerate(histories):
            pieces = [sid_pair[modality_index] for sid_pair in history]
            sequence = list(chain.from_iterable(pieces)) + [self.eos_id()]
            sequence = torch.tensor(sequence, dtype=torch.long)
            if self.config.padding == "left":
                input_ids[row, -len(sequence):] = sequence
            elif self.config.padding == "right":
                input_ids[row, :len(sequence)] = sequence
            else:
                raise ValueError("padding must be either 'left' or 'right'")
        return input_ids, (input_ids != self.pad_id()).to(torch.float)

    def get_user_len(self):
        return len(self.umap)
    
    def get_users(self):
        return list(self.umap.values())
    
    def get_item_len(self):
        return len(self.smap)

    def get_items(self):
        return list(self.smap.values())

    def pad_id(self):
        return self.config.pad_token_id
    
    def eos_id(self):
        return self.config.eos_token_id


class SBRPtDataset(absSBRDataset):
    def __init__(self, args):
        super().__init__(args)
        self.train_json = json.load(open(path.join(self.config.data_dir, self.config.dataset, "train.json")))        
        self._process()
    
    def _process(self):
        self.train_log = list(self.val_json.values())
        if self.config.slide_window is False:
            self.train_log = list(self.train_json.values())
        elif self.config.slide_window is True:
            raw_logs = self.train_log
            expanded_logs = []
            for logs in raw_logs:
                if len(logs) < 2:
                    continue
                else:
                    for offset in range(2, len(logs)+1):
                        expanded_logs.append(logs[:offset])
            self.train_log = expanded_logs

    def __len__(self):
        return len(self.train_log)
    
    def __getitem__(self, index):
        item_history = self.train_log[index][-(self.config.max_history_length):]
        return [self.item_sid_pair(item) for item in item_history]
    
    def collate(self, batch: list[list]) -> list[torch.Tensor, torch.Tensor, list[int]]:
        histories = [sample[:-1] for sample in batch]
        text_input_ids, text_attention_mask = self.collate_history(histories, 0)
        image_input_ids, image_attention_mask = self.collate_history(histories, 1)
        text_labels = torch.tensor(
            [sample[-1][0] + [self.eos_id()] for sample in batch],
            dtype=torch.long,
        )
        image_labels = torch.tensor(
            [sample[-1][1] + [self.eos_id()] for sample in batch],
            dtype=torch.long,
        )
        return [
            text_input_ids,
            text_attention_mask,
            image_input_ids,
            image_attention_mask,
            text_labels,
            image_labels,
        ]


class SBRValDataset(absSBRDataset):
    def __init__(self, args):
        super().__init__(args)
        self.train_json = json.load(open(path.join(self.config.data_dir, self.config.dataset, "train.json")))
        self.val_json = json.load(open(path.join(self.config.data_dir, self.config.dataset, "val.json")))
        self.test_json = json.load(open(path.join(self.config.data_dir, self.config.dataset, "test.json")))
        self._process()
    
    def _process(self):
        self.train_log = list(self.train_json.values())
        self.val_log = list(self.val_json.values())

    def __len__(self):
        return len(self.train_json)
    
    def __getitem__(self, index):
        item_history = self.train_log[index][-(self.config.max_history_length):]
        history = [self.item_sid_pair(item) for item in item_history]
        return [history, self.val_log[index][0]]
    
    def collate(self, batch) -> list[dict, torch.Tensor]:
        histories = [sample[0] for sample in batch]
        text_input_ids, text_attention_mask = self.collate_history(histories, 0)
        image_input_ids, image_attention_mask = self.collate_history(histories, 1)
        target = torch.tensor(
            [sample[1] for sample in batch],
            dtype=torch.long,
        )
        return [
            text_input_ids,
            text_attention_mask,
            image_input_ids,
            image_attention_mask,
            target,
        ]

class SBRTestDataset(absSBRDataset):
    def __init__(self, args):
        super().__init__(args)
        self.train_json = json.load(open(path.join(self.config.data_dir, self.config.dataset, "train.json")))
        self.val_json = json.load(open(path.join(self.config.data_dir, self.config.dataset, "val.json")))
        self.test_json = json.load(open(path.join(self.config.data_dir, self.config.dataset, "test.json")))
        self._process2()

    def _process2(self):
        self.train_log = [a + b for a, b in zip(list(self.train_json.values()), list(self.val_json.values()))]
        self.val_log = list(self.test_json.values())
    
    def __len__(self):
        return len(self.train_json)
    
    def __getitem__(self, index):
        item_history = self.train_log[index][-(self.config.max_history_length):]
        history = [self.item_sid_pair(item) for item in item_history]
        return [history, self.val_log[index][0]]
    
    def collate(self, batch) -> list[dict, torch.Tensor]:
        histories = [sample[0] for sample in batch]
        text_input_ids, text_attention_mask = self.collate_history(histories, 0)
        image_input_ids, image_attention_mask = self.collate_history(histories, 1)
        target_items = torch.tensor(
            [sample[1] for sample in batch],
            dtype=torch.long,
        )
        return [
            text_input_ids,
            text_attention_mask,
            image_input_ids,
            image_attention_mask,
            target_items,
        ]


class SBRDataModule(L.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.config = args
    
    def setup(self, stage=None):
        if stage == 'fit':
            self.train_ds = SBRPtDataset(self.config)
            self.val_ds = SBRValDataset(self.config)
            self.test_ds = SBRTestDataset(self.config)
        elif stage == 'validate':
            self.val_ds = SBRValDataset(self.config)
        elif stage == 'test':
            self.test_ds = SBRTestDataset(self.config)
        elif stage == 'predict':
            self.test_ds = SBRTestDataset(self.config)
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.batch_size, 
                          shuffle=True, collate_fn=self.train_ds.collate, num_workers=self.config.num_workers,
                          persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config.eval_batch_size, 
                          collate_fn=self.val_ds.collate)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.config.eval_batch_size, 
                          collate_fn=self.test_ds.collate)
    
    def predict_dataloader(self):
         return DataLoader(self.test_ds, batch_size=self.config.eval_batch_size, 
                          collate_fn=self.test_ds.collate)


class EmbDataset(Dataset):

    def __init__(self, args):
        super().__init__()
        self.config = args
        self.embeddings_v = np.load(path.join(self.config.data_dir, self.config.dataset, f'{self.config.sid_group_v}_embeddings.npy'))
        self.embeddings_v = torch.from_numpy(self.embeddings_v)[:, :self.config.plm_emb_dim]
        self.embeddings_t = np.load(path.join(self.config.data_dir, self.config.dataset, f'{self.config.sid_group_t}_embeddings.npy'))
        self.embeddings_t = torch.from_numpy(self.embeddings_t)[:, :self.config.plm_emb_dim]

    def __len__(self):
        return len(self.embeddings_v)
    
    def __getitem__(self, index):
        return self.embeddings_t[index], self.embeddings_v[index]
    


class SemIDDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

    def prepare_data(self) -> None:
        folder = Path(path.join(self.config.data_dir, self.config.dataset))
        designated_files = ['train.json', 'val.json', 'test.json', 'smap.json', 'umap.json', 
                            f'{self.config.sid_group_v}_embeddings.npy', f'{self.config.sid_group_t}_embeddings.npy']
        if not folder.exists() or self.config.force_preprocess:
            preprocess(self.config)
        else:
            for file in designated_files:
                if not (folder / file).exists():
                    logger.error(f"{file} is missing. Re-run preprocessing.")
                    preprocess(self.config)
                    break
                    
    
    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_ds = EmbDataset(self.config)
            self.val_ds = EmbDataset(self.config)
        if stage == 'validate':
            self.val_ds = EmbDataset(self.config)
        if stage == 'predict':
            self.test_ds = EmbDataset(self.config)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.rq_batch_size, 
                          shuffle=True, num_workers=self.config.num_workers,
                          persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config.rq_batch_size, 
                          shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.config.rq_batch_size, 
                          shuffle=False)

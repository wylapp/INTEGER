import json
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm


def preprocess(cfg):
    meta_ds = pd.read_parquet(os.path.join(cfg.data_dir, cfg.dataset, "meta.parquet"))

    model = SentenceTransformer(
        cfg.vlm,
        model_kwargs={"dtype": torch.bfloat16}
    )
    flat_items = []
    for _, item in tqdm(meta_ds.iterrows()):
        flat_items.append({
            "image": os.path.join(cfg.data_dir, cfg.dataset, "cards", f"{item['asin']}.jpg")
            })

    with torch.no_grad():
        embeddings = model.encode(flat_items, 
                                  batch_size=32,
                                  prompt="Represent this product for semantic similarity and recommendation.",
                                  show_progress_bar=True, 
                                  convert_to_numpy=True)

    np.save(os.path.join(cfg.data_dir, cfg.dataset, f"{cfg.sid_group_v}_embeddings.npy"), embeddings)

    return True

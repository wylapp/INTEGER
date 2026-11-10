import gzip
import json
import os
from numbers import Number

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils import word_tokenize


UNAVAILABLE = "Data Unavailable"
MIN_USER_INTERACTIONS = 5
META_COLUMNS = [
    "index",
    "asin",
    "description",
    "title",
    "categories",
    "price",
    "brand",
    "itemID",
]


def parse(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def getDF(path):
    return list(parse(path))


def _as_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return UNAVAILABLE
    if isinstance(value, (list, tuple)):
        parts = [_as_text(part) for part in value]
        return " ".join(part for part in parts if part != UNAVAILABLE) or UNAVAILABLE
    text = str(value).strip()
    return text or UNAVAILABLE


def _format_categories(value):
    if not isinstance(value, (list, tuple)) or not value:
        return _as_text(value)
    branch = value[0] if isinstance(value[0], (list, tuple)) else value
    categories = [_as_text(category) for category in branch]
    return ">".join(category for category in categories if category != UNAVAILABLE) or UNAVAILABLE


def _format_price(value):
    if isinstance(value, Number) and not isinstance(value, bool) and np.isfinite(value):
        return f"${value:.2f}"
    return _as_text(value)


def _has_title(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(_has_title(part) for part in value)
    return False


def _has_high_res_image(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return bool(value) and isinstance(value[0], str) and bool(value[0].strip())
    return False


def _prepare_metadata(meta_ds, smap, desc_maxlen):
    required_columns = {"asin", "title", "imageURLHighRes"}
    missing_columns = required_columns.difference(meta_ds.columns)
    if missing_columns:
        raise ValueError(f"Amazon metadata is missing required fields: {sorted(missing_columns)}")

    meta_ds = meta_ds.reset_index()
    meta_ds = meta_ds.loc[
        meta_ds["asin"].isin(smap)
        & meta_ds["title"].apply(_has_title)
        & meta_ds["imageURLHighRes"].apply(_has_high_res_image),
        :,
    ].copy()
    meta_ds.drop_duplicates(subset=["asin"], keep="first", inplace=True)
    meta_ds.set_index("asin", inplace=True, drop=False)
    meta_ds = meta_ds.reindex(list(smap))

    missing = meta_ds["asin"].isna()
    if missing.any():
        raise ValueError(
            "Metadata with a non-empty title and high-resolution image URL is "
            f"unavailable for {missing.sum()} mapped items"
        )

    if "categories" not in meta_ds and "category" in meta_ds:
        meta_ds["categories"] = meta_ds["category"]
    if "categories" in meta_ds and "main_cat" in meta_ds:
        empty_categories = meta_ds["categories"].apply(
            lambda value: value is None
            or value == ""
            or (isinstance(value, (list, tuple)) and not value)
        )
        meta_ds.loc[empty_categories, "categories"] = meta_ds.loc[
            empty_categories, "main_cat"
        ]
    for column in ("description", "title", "categories", "price", "brand"):
        if column not in meta_ds:
            meta_ds[column] = UNAVAILABLE

    meta_ds["description"] = meta_ds["description"].apply(
        lambda value: " ".join(word_tokenize(_as_text(value))[:desc_maxlen])
    )
    meta_ds["title"] = meta_ds["title"].apply(_as_text)
    meta_ds["categories"] = meta_ds["categories"].apply(_format_categories)
    meta_ds["price"] = meta_ds["price"].apply(_format_price)
    meta_ds["brand"] = meta_ds["brand"].apply(_as_text)
    meta_ds["itemID"] = meta_ds["asin"].map(smap).astype("int64")
    meta_ds["index"] = meta_ds["index"].fillna(-1).astype("int64")

    return meta_ds.sort_values("itemID").reset_index(drop=True)[META_COLUMNS]


def preprocess(cfg):
    output_dir = os.path.join(cfg.data_dir, cfg.dataset)
    os.makedirs(output_dir, exist_ok=True)

    review_path = os.path.join(cfg.data_dir, f"{cfg.dataset}_5.json.gz")
    metadata_path = os.path.join(cfg.data_dir, f"meta_{cfg.dataset}.json.gz")
    if not os.path.isfile(review_path):
        raise FileNotFoundError(
            f"Amazon Review 2018 5-core file not found: {review_path}"
        )
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Amazon Review 2018 metadata file not found: {metadata_path}")

    reviewpd = pd.DataFrame(getDF(review_path))
    required_review_columns = {"reviewerID", "asin", "unixReviewTime"}
    missing_columns = required_review_columns.difference(reviewpd.columns)
    if missing_columns:
        raise ValueError(f"Amazon reviews are missing required fields: {sorted(missing_columns)}")

    meta_ds = pd.DataFrame(getDF(metadata_path))
    required_metadata_columns = {"asin", "title", "imageURLHighRes"}
    missing_columns = required_metadata_columns.difference(meta_ds.columns)
    if missing_columns:
        raise ValueError(f"Amazon metadata is missing required fields: {sorted(missing_columns)}")

    eligible_metadata = meta_ds["title"].apply(_has_title) & meta_ds[
        "imageURLHighRes"
    ].apply(_has_high_res_image)
    eligible_asins = set(meta_ds.loc[eligible_metadata, "asin"])
    original_items = reviewpd["asin"].nunique()
    reviewpd = reviewpd.loc[reviewpd["asin"].isin(eligible_asins), :].copy()
    logger.info(
        f"Removed {original_items - reviewpd['asin'].nunique()} items without both "
        "a title and a high-resolution image URL"
    )

    interaction_counts = reviewpd.groupby("reviewerID")["asin"].size()
    retained_users = set(
        interaction_counts[interaction_counts >= MIN_USER_INTERACTIONS].index
    )
    removed_users = reviewpd["reviewerID"].nunique() - len(retained_users)
    reviewpd = reviewpd.loc[reviewpd["reviewerID"].isin(retained_users), :].copy()
    if reviewpd.empty:
        raise ValueError("No user sequences remain after metadata filtering")
    if removed_users:
        logger.info(
            f"Removed {removed_users} users with fewer than "
            f"{MIN_USER_INTERACTIONS} remaining interactions"
        )

    logger.info("Generating numerical indices")
    umap = {user: index for index, user in enumerate(reviewpd["reviewerID"].unique())}
    smap = {item: index for index, item in enumerate(reviewpd["asin"].unique())}

    reviewpd["reviewerID"] = reviewpd["reviewerID"].map(umap)
    reviewpd["asin"] = reviewpd["asin"].map(smap)
    user_interactions = (
        reviewpd.sort_values("unixReviewTime", kind="stable")
        .groupby("reviewerID", sort=True)["asin"]
        .apply(list)
        .reset_index()
    )

    logger.info(f"#items: {len(smap)} #users: {len(umap)}")

    train_seq = {}
    val_seq = {}
    test_seq = {}
    for _, interactions in tqdm(
        user_interactions.iterrows(),
        total=len(user_interactions),
        desc="Generating train/val/test split",
    ):
        user = str(interactions["reviewerID"])
        items = interactions["asin"]
        train_seq[user] = items[:-2]
        val_seq[user] = [items[-2]]
        test_seq[user] = [items[-1]]

    if any(len(items) < MIN_USER_INTERACTIONS - 2 for items in train_seq.values()):
        raise ValueError("Training sequences must contain at least 3 interactions")

    outputs = {
        "train.json": train_seq,
        "val.json": val_seq,
        "test.json": test_seq,
        "smap.json": smap,
        "umap.json": umap,
    }
    for filename, content in outputs.items():
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as stream:
            json.dump(content, stream)

    logger.info("Sequences are ready. Proceeding to metadata parsing.")
    valid_ds = _prepare_metadata(meta_ds, smap, cfg.desc_maxlen)
    valid_ds.to_parquet(os.path.join(output_dir, "meta.parquet"))

    logger.info("Metadata is ready. Now preparing embeddings.")
    model = SentenceTransformer(
        cfg.plm,
        # device="cuda:1",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    flat_items = [
        f"ID: {item.itemID}; Title: {item.title}; Category: {item.categories}; "
        f"Brand: {item.brand}; Description: {item.description}"
        for item in tqdm(valid_ds.itertuples(index=False), total=len(valid_ds))
    ]

    with torch.no_grad():
        embeddings = model.encode(
            flat_items,
            batch_size=cfg.encode_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    emb_group = getattr(cfg, "sid_group_t", "plm")
    np.save(os.path.join(output_dir, f"{emb_group}_embeddings.npy"), embeddings)

    return True

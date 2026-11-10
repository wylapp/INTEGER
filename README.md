# INTEGER: Interaction-Mediated Multimodal Item Tokenization for Generative Recommendation

## Usage

1. Download the raw data as described in [`data_18`](data_18/readme.md).
2. Run `python prepare_data.py` to preprocess the data.
3. Follow the instructions in [`VI`](VI/readme.md) to render the visual interface for each item.
4. Run `python prepare_data_mm.py` to generate semantic embeddings for both modalities.
5. Run `python main_irc.py` to train the IRC module and generate calibrated item representations.
6. Run `python main_r2vq.py` to train the quantizer and generate semantic IDs (SIDs) for all items.
7. Run `python main_seq2seq.py` to train and evaluate the final generative recommender.

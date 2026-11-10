# Visual Interface Rendering

Product images are not distributed with this repository. After preprocessing, manually download the first image listed in the `imageURLHighRes` field for each item. Save each image as `<asin>.jpg` in the processed dataset's `images` directory, and ensure that your use of the images complies with the Amazon's Terms of Use.

For example:

```
data_18
├── Arts_Crafts_and_Sewing
│   ├── images
│   │   ├── B001390CK0.jpg
```

Set `DATA_FOLDER` and `DATASET` in `render_cards.py` to match the processed data. Then run the following command from this directory:

```bash
python render_cards.py
```

# Deep FiLM Image-Text Model

This project implements a Deep FiLM (Feature-wise Linear Modulation) model for image-text tasks. It builds upon the Universal CLIP architecture, using film-based modules to modulate image embeddings at each decoder with embeddings from the text-based decoder.

## Setup

1. Install the required environment:
   ```
   conda env create -f environment.yml
   ```

2. Download the dataset as specified in the Universal CLIP paper.

## Training

To train the model, use the `train_deep_film.py` script:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train_deep_film.py --dist True --uniform_sample
```

The training recipe is similar to that described in the Universal CLIP paper.

## Evaluation

To evaluate the model, use the `model_test_try.py` script:

```
python model_test_try.py --weight <path_to_weights> --model_type film
```

Replace `<path_to_weights>` with the path to your trained model weights, and `<model_type>` with either "UniversalCLIP" or "DeepFiLM".

### Preparing Test Data

Before evaluation, you need to create a test dataset file:

1. Create a file named `test_dataset.txt`.
2. In this file, list the locations of the files in your text directory.
3. Format each line as follows:
   ```
   file_path<tab>label
   ```
   Note: If you don't have labels, you can repeat the same file_path after the tab space for each entry.

Name your test dataset files as `<name>_train.txt` or `<name>_val.txt`.

## Saving Results

To save your prediction results, use the `save_result` function from `gg_tools.py`.

## Notes

- This project is based on the Universal CLIP architecture with modifications.
- The Deep FiLM model modulates image embeddings using text-based decoder outputs.
- Ensure you have the necessary computational resources for training and evaluation.

For more detailed information on the Universal CLIP architecture, please refer to the original paper.

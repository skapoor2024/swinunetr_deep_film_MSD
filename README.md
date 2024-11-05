# Deep FiLM Image-Text Model

This project implements a CLIP-Deep-Driven FiLM and CLIP-Deep-FIlM-2 (Feature-wise Linear Modulation) model for image-text tasks. It builds upon the Universal CLIP architecture, using film-based modules to modulate image embeddings at each decoder with embeddings from the text-based decoder.

## CLIP-Deep-Driven Architecture
The architecture for train_deep_film.py
```mermaid
flowchart TD
   Input[Input Image] --> SwinViT[SwinViT Transformer]
   Input --> Enc0[Encoder 1]
   
   subgraph Text_Processing
      Prompt[Text Prompt] --> CLIP[CLIP Text Encoder]
      CLIP --> TextEmb[Text Embeddings]
   end

   SwinViT --> HS0[Hidden State 0]
   SwinViT --> HS1[Hidden State 1]
   SwinViT --> HS2[Hidden State 2]
   SwinViT --> HS3[Hidden State 3]
   SwinViT --> HS4[Hidden State 4]

   HS0 --> Enc1[Encoder 2]
   HS1 --> Enc2[Encoder 3]
   HS2 --> Enc3[Encoder 4]
   HS4 --> Dec4[Encoder 10]

   Dec4 --> FilmBlock4[FiLM Block 4]
   TextEmb --> FilmBlock4
   
   FilmBlock4 --> Dec3[Decoder 5]
   HS3 --> Dec3

   Dec3 --> FilmBlock3[FiLM Block 3]
   TextEmb --> FilmBlock3
   
   FilmBlock3 --> Dec2[Decoder 4]
   Enc3 --> Dec2

   Dec2 --> FilmBlock2[FiLM Block 2]
   TextEmb --> FilmBlock2
   
   FilmBlock2 --> Dec1[Decoder 3]
   Enc2 --> Dec1

   Dec1 --> FilmBlock1[FiLM Block 1]
   TextEmb --> FilmBlock1
   
   FilmBlock1 --> Dec0[Decoder 2]
   Enc1 --> Dec0

   Dec0 --> FinalDec[Decoder 1]
   Enc0 --> FinalDec

   FinalDec --> Output[Output Logits]

   style Input fill:#f9f,stroke:#333,stroke-width:2px
   style Output fill:#9ff,stroke:#333,stroke-width:2px
   style TextEmb fill:#ff9,stroke:#333,stroke-width:2px
   
```

## Deep_film_2 Architecture

```mermaid

flowchart TD
   Input[Input Image] --> SwinViT[SwinViT Transformer]
   Input --> Enc0[Encoder 1]
   
   subgraph Text_Processing
      Prompt[Text Prompt] --> CLIP[CLIP Text Encoder]
      CLIP --> TextEmb[Text Embeddings]
   end

   SwinViT --> HS0[Hidden State 0]
   SwinViT --> HS1[Hidden State 1]
   SwinViT --> HS2[Hidden State 2]
   SwinViT --> HS3[Hidden State 3]
   SwinViT --> HS4[Hidden State 4]

   HS0 --> Enc1[Encoder 2]
   Enc1 --> FilmEnc1[FiLM Block 0]
   TextEmb --> FilmEnc1

   HS1 --> Enc2[Encoder 3]
   Enc2 --> FilmEnc2[FiLM Block 1]
   TextEmb --> FilmEnc2

   HS2 --> Enc3[Encoder 4]
   Enc3 --> FilmEnc3[FiLM Block 2]
   TextEmb --> FilmEnc3

   HS3 --> FilmEnc4[FiLM Block 3]
   TextEmb --> FilmEnc4

   HS4 --> Dec4[Encoder 10]
   Dec4 --> FilmDec4[FiLM Block 4]
   TextEmb --> FilmDec4

   FilmDec4 --> Dec3[Decoder 5]
   FilmEnc4 --> Dec3

   Dec3 --> Dec2[Decoder 4]
   FilmEnc3 --> Dec2

   Dec2 --> Dec1[Decoder 3]
   FilmEnc2 --> Dec1

   Dec1 --> Dec0[Decoder 2]
   FilmEnc1 --> Dec0

   Dec0 --> FinalDec[Decoder 1]
   Enc0 --> FinalDec

   FinalDec --> Output[Output Logits]

   style Input fill:#f9f,stroke:#333,stroke-width:2px
   style Output fill:#9ff,stroke:#333,stroke-width:2px
   style TextEmb fill:#ff9,stroke:#333,stroke-width:2px

```

## Setup

The required environment files are present inside the env_file folder.

1. Install the required environment:
   ```
   conda env create -f environment.yml
   ```

2. Download the dataset as specified in the Universal CLIP paper.

## Training

Training
For each dataset create DatasetName_train.txt DatasetName_test.txt DatasetName_val.txt under the /dataset/datasetlist folder

The model will look out for the training dataset list for DatasetName that contains the training files like {img,’\t’,seg} _train.txt and _val.txt important to get the training started associated with the data

To train the model, use the `train_deep_film.py` or `train_deep_film_2.py` script:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train_deep_film.py --dist True --uniform_sample
```

The training recipe is similar to that described in the Universal CLIP paper.

## Distributed Training (For deep_film_2_multinode)

```
srun python train_deep_film_2_multinode.py \
    --dist True \
    --uniform_sample \
    --num_workers 4 \
    --log_name deep_film_2_multinode_og_setting \
    --world_size $WORLD_SIZE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT

```

## Evaluation

To evaluate the model, use the `model_test.py` and `model_val.py` script:
For both test and val the <img,label> pair location must be there in the datase

```
Model_val.py —pretrain <your_pretrain_folder> –model_type <model_name> –file_name <file_name_for_result> –datasetlist <txt file containing image_seg_pair of datalist>
```

Note:- At datasetlist just type the name of the dataset.


Model_name = universal, swinunetr, unetr, film 

```
Model_test.py —pretrain <your_pretrain_folder> –model_type <model_name> –file_name <file_name_for_result> -–datasetlist <txt file containing image_seg_pair of datalist>
```

To create data list
If PAOT is the dataset then PAOT_train contains the {img,’\t’,seg}  type and similarly for PAOT_test and PAOT_val

Just write the DatasetName with argumen –datasetlist 

Example Scirpts

```
python model_test.py --pretrain ./out/unetr_monai/epoch_160.pth --model_type unetr --file_name unetr_paot_btcv_test_e160.txt --dataset_list btcv
```


```
python model_val.py --pretrain ./out/unetr_monai/epoch_160.pth --model_type unetr --file_name unetr_paot_btcv_e160.txt --dataset_list btcv3
```

### Preparing Test Data

Before evaluation, you need to create a test dataset file:

1. Create a file named `dataset_text.txt`.
2. In this file, list the locations of the files in your text directory.
3. Format each line as follows:
   ```
   file_path<tab>label
   ```

## Saving Results

To save your prediction results, use the `save_result` function from `gg_tools.py`.
Or you can use `model_test_save_predictions_final.py`.
Here the loader will take `_test2.txt` files. These files dont have their ground truths. So inside datasetname_test2.txt files, it should contain the names of the files to be predicted

## Some predictions

Predictions of train_deep_film or CLIP-DEEP-Driven Model

![12-CT-ORG][images/13_AbdomenCT-12organ_label_Organ12_0020_axial_slice_106_comparison 1.png]
![KiTS][images/05_KiTS_label_label0071_axial_slice_161_comparison 1.png]
![AMOS][images/09_AMOS_label_amos_0111_axial_slice_305_comparison 1.png]

## Notes

- This project is based on the Universal CLIP architecture with modifications.
- The Deep FiLM model modulates image embeddings using text-based decoder outputs.
- Ensure you have the necessary computational resources for training and evaluation.
- train_deep_film_2 contains fixes for abritrary text input
- train_deep_film_2 uses film module with pre residual normalization and residual layer.
- train_val train files are with different custom data loader than normal train files. The noraml train files brought more accuracy due to uniform sampling

For more detailed information on the Universal CLIP architecture, please refer to the original paper.

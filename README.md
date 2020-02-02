# ADORE_Generator

This project aims to generate aspect-aware reviews. 

Requiement: pytorch 1.4

Download pre-trained restaurant word embeddings from https://drive.google.com/uc?id=12Pj5LkKnE_XQKIRABiviqgspA5DB1Zfn&export=download

To train the model run:

time CUDA_VISIBLE_DEVICES=0 python main.py --inputt ./input  --save_1 model_rnn.pt --save_2 model_context.pt

To generate aspect-aware reviews run:

time python generate.py

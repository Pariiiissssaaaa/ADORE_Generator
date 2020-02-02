# ADORE_Generator

This project aims to generate aspect-aware reviews. pytorch 1.4
To train the model run:

time CUDA_VISIBLE_DEVICES=0 python main.py --inputt ./input  --save_1 model_rnn.pt --save_2 model_context.pt

To generate aspect-aware reviews run:

time python generate.py

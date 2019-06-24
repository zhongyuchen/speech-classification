export CUDA_VISIBLE_DEVICES=0,1,2

python run_cnn.py --model cnn --data mfcc

#python run_cnn.py --model cnn --data spectrogram

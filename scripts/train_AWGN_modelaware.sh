# AWGN model aware

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_1

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=500 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_2

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=1000 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_3

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-2 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_4

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=500 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-2 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_5

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=1000 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-2 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_6

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-4 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_7

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=500 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-4 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_8

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=1000 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-4 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_9

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 3 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_10

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=500 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 3 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_11

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=1000 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 3 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_12

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-2 --optimizer adam --md_reg 1e-3 --train_ratio 3 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_13

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=500 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-2 --optimizer adam --md_reg 1e-3 --train_ratio 3 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_14

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=1000 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-2 --optimizer adam --md_reg 1e-3 --train_ratio 3 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_15

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-4 --optimizer adam --md_reg 1e-3 --train_ratio 3 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_16

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=500 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-4 --optimizer adam --md_reg 1e-3 --train_ratio 3 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_17

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=1000 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-4 --optimizer adam --md_reg 1e-3 --train_ratio 3 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_18

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 7 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_19

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=500 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 7 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_20

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=1000 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 7 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_21

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-2 --optimizer adam --md_reg 1e-3 --train_ratio 7 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_22

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=500 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-2 --optimizer adam --md_reg 1e-3 --train_ratio 7 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_23

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=1000 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-2 --optimizer adam --md_reg 1e-3 --train_ratio 7 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_24

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-4 --optimizer adam --md_reg 1e-3 --train_ratio 7 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_25

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=500 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-4 --optimizer adam --md_reg 1e-3 --train_ratio 7 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_26

python train.py --SNR 5 --md_len=6 --channel=AWGN --batch_size=1000 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-4 --optimizer adam --md_reg 1e-3 --train_ratio 7 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --name AWGN_modelaware_27

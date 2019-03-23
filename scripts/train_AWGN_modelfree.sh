# model free

# AWGN
# SNR 1
python train.py --modelfree --SNR 1 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_1_1

python train.py --modelfree --SNR 1 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_1_2

python train.py --modelfree --SNR 1 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_1_3

python train.py --modelfree --SNR 1 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_1_4

python train.py --modelfree --SNR 1 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_1_5

# SNR 2
python train.py --modelfree --SNR 2 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_2_1

python train.py --modelfree --SNR 2 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_2_2

python train.py --modelfree --SNR 2 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_2_3

python train.py --modelfree --SNR 2 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_2_4

python train.py --modelfree --SNR 2 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_2_5

# SNR 3
python train.py --modelfree --SNR 3 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_3_1

python train.py --modelfree --SNR 3 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_3_2

python train.py --modelfree --SNR 3 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_3_3

python train.py --modelfree --SNR 3 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_3_4

python train.py --modelfree --SNR 3 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_3_5

# SNR 4
python train.py --modelfree --SNR 4 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_4_1

python train.py --modelfree --SNR 4 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_4_2

python train.py --modelfree --SNR 4 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_4_3

python train.py --modelfree --SNR 4 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_4_4

python train.py --modelfree --SNR 4 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_4_5

# SNR 5
python train.py --modelfree --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_5_1

python train.py --modelfree --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_5_2

python train.py --modelfree --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_5_3

python train.py --modelfree --SNR 5 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_5_4

python train.py --modelfree --SNR 45 --md_len=6 --channel=AWGN --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name AWGN_modelfree_SNR_5_5

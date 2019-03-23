# model free

# BEC
# adjust epsilon and sigma
python train.py --epsilon .05 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.05 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_05_0

python train.py --epsilon .05 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_05_1

python train.py --epsilon .05 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_05_2

python train.py --epsilon .05 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_05_3

python train.py --epsilon .05 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_05_4

python train.py --epsilon .05 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_05_5

python train.py --epsilon .05 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_05_6

# eps = 0.1

python train.py --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_1

python train.py --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_2

python train.py --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_3

python train.py --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_4

python train.py --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_5

python train.py --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_6

# eps = 0.2

python train.py --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_1

python train.py --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_2

python train.py --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_3

python train.py --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_4

python train.py --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_5

python train.py --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_6

# eps = 0.3

python train.py --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_1

python train.py --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_2

python train.py --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_3

python train.py --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_4

python train.py --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_5

python train.py --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_6

# eps = 0.4

python train.py --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_1

python train.py --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_2

python train.py --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_3

python train.py --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_4

python train.py --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_5

python train.py --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_6

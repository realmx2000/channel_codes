# model free

# BEC
# adjust epsilon and sigma

# eps = .1

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_1

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_2

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_3

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_4

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_5

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_1_6

# eps = .2

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_1

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_2

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_3

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_4

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_5

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_2_6

# eps = .3

python train.py --modelfree --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_1

python train.py --modelfree --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_2

python train.py --modelfree --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_3

python train.py --modelfree --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_4

python train.py --modelfree --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_5

python train.py --modelfree --epsilon .3 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_3_6

# eps = .4

python train.py --modelfree --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_1

python train.py --modelfree --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_2

python train.py --modelfree --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_3

python train.py --modelfree --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_4

python train.py --modelfree --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_5

python train.py --modelfree --epsilon .4 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_4_6

# eps = .5

python train.py --modelfree --epsilon .5 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_5_1

python train.py --modelfree --epsilon .5 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_5_2

python train.py --modelfree --epsilon .5 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_5_3

python train.py --modelfree --epsilon .5 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_5_4

python train.py --modelfree --epsilon .5 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_5_5

python train.py --modelfree --epsilon .5 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_5_6

# eps = .6

python train.py --modelfree --epsilon .6 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_6_1

python train.py --modelfree --epsilon .6 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_6_2

python train.py --modelfree --epsilon .6 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_6_3

python train.py --modelfree --epsilon .6 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_6_4

python train.py --modelfree --epsilon .6 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_6_5

python train.py --modelfree --epsilon .6 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_6_6

# eps = .7

python train.py --modelfree --epsilon .7 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_7_1

python train.py --modelfree --epsilon .7 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_7_2

python train.py --modelfree --epsilon .7 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_7_3

python train.py --modelfree --epsilon .7 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_7_4

python train.py --modelfree --epsilon .7 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_7_5

python train.py --modelfree --epsilon .7 --md_len=6 --channel=BEC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BEC_modelfree_eps_7_6


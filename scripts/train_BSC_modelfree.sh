# model free

# BSC
# adjust epsilon and sigma
python train.py --modelfree --epsilon .05 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.05 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_05_0

python train.py --modelfree --epsilon .05 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_05_1

python train.py --modelfree --epsilon .05 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_05_2

python train.py --modelfree --epsilon .05 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_05_3

python train.py --modelfree --epsilon .05 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_05_4

python train.py --modelfree --epsilon .05 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_05_5

python train.py --modelfree --epsilon .05 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_05_6

# eps = 0.1

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_10_1

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_10_2

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_10_3

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_10_4

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_10_5

python train.py --modelfree --epsilon .1 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_10_6

# eps 0.15

python train.py --modelfree --epsilon .15 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_15_1

python train.py --modelfree --epsilon .15 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_15_2

python train.py --modelfree --epsilon .15 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_15_3

python train.py --modelfree --epsilon .15 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_15_4

python train.py --modelfree --epsilon .15 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_15_5

python train.py --modelfree --epsilon .15 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_15_6

# eps = 0.2

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_20_1

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_20_2

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_20_3

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_20_4

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_20_5

python train.py --modelfree --epsilon .2 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_20_6

# eps = 0.25

python train.py --modelfree --epsilon .25 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.1 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_25_1

python train.py --modelfree --epsilon .25 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.2 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_25_2

python train.py --modelfree --epsilon .25 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.3 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_25_3

python train.py --modelfree --epsilon .25 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.4 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_25_4

python train.py --modelfree --epsilon .25 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.5 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_25_5

python train.py --modelfree --epsilon .25 --md_len=6 --channel=BSC --batch_size=100 --redundancy=3 --sigma=0.6 --block_length 100 --loss bce --lr 1e-3 --optimizer adam --md_reg 1e-3 --train_ratio 5 --enc_size 50 --dec_size 100 --layers 2 --sigma_decay=0.99 --gpu --name BSC_modelfree_eps_25_6

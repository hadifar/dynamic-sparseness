# Evaluation

Code to reproduce our results.

Do the following steps to get results on table 2:

- Install `gmul`
- Install [distiller](https://github.com/IntelLabs/distiller#installation)
- run the following command

Baseline LSTM:

- `python3 main.py --data data/ptb/ --cuda --model LSTMCell -- emsize 1536 --nhid 1536 --dropout 0.65 --batch_size 128 --wd 0`

Baseline AGP:
- `python3 main.py --data data/ptb/ --cuda --compress word_lang_model.LARGE_50.schedule_agp.yaml --model LSTMCell -- emsize 1536 --nhid 1536 --dropout 0.65 --batch_size 128 --wd 0`

Baseline static-block:
- `python3 main.py --data data/ptb/ --cuda --model GatedLSTMCell --mode static --emsize 1536 --nhid 1536 --dropout 0.65 --batch_size 128 --wd 0`

Dynamic sparseness:

- `python3 main.py --data data/ptb/ --cuda --model GatedLSTMCell --mode dynamic --emsize 1536 --nhid 1536 --dropout 0.65 --batch_size 128 --wd 0 --sp 0.5` 


Note that tuning weight sharing (`--tied`), weight decay (`--wd`) and dropout (`--dropout`) hyper-parameters can produce slightly better results.
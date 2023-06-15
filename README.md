## Result on dataset
We summarize the validation results as follows.


|Name              |Accuracy|Precision|Recall|F1    |Specificity| Weight|
|--------------------|--------|---------|------|------|-----------|-----------|
|Backbone            |98.254  |0.9773   |0.98  |0.9785|0.992      | [download](https://github.com/xwj260817/covid-effnet/releases/download/v0.1/covid_effnet.pth) |
|Backbone + SFD        |98.316  |0.9788   |0.9797|0.9792|0.9922     | [download](https://github.com/xwj260817/covid-effnet/releases/download/v0.1/covid_effnet_sfd.pth) |
|Backbone + Attack    |98.667  |0.9836   |0.9839|0.9837|0.9938     | [download](https://github.com/xwj260817/covid-effnet/releases/download/v0.1/covid_effnet_attack.pth) |
|Backbone + SFD + Attack|98.702  |0.9837   |0.9846|0.9841|0.994      | [download](https://github.com/xwj260817/covid-effnet/releases/download/v0.1/covid_effnet_sfd_attack.pth) |

## Example: Train model
train on single gpu:
```bash
python train.py ./dataset/ --model covid_effnet \
--sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 \
-j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.4 --drop-path 0.2 \
--amp --lr .048 \
--num-classes 3 --native-amp -b 16 \
--attack-iter 1 --attack-epsilon 1 --attack-step-size 1 --mixbn
```

train on multi gpu:
```bash
./dist_train.sh 8 ./dataset/ --model covid_effnet \
--sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 \
-j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.4 --drop-path 0.2 --amp --lr .048 \
--num-classes 3 --native-amp -b 32 \
--attack-iter 1 --attack-epsilon 1 --attack-step-size 1 --mixbn
```

## Citation


### Contact Information

For help or issues using models, please submit a GitHub issue.

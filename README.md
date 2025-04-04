## ongoing

### Why vocos with transformer or conformer ?
Easy to scale and Good control over latency and caching

### Why sequence mask in gan?
No data length limit is required, such as 1s

###  Why in wenet ？
cache and multiple speech models are  available out of the box


### Data Prepare
```bash
{"wav": "/data/BAC009S0764W0121.wav"}
{"wav": "/data/BAC009S0764W0122.wav"}
```
### train
```bash
train_data = 'train.jsonl'
model_dir = 'vocos/exp/2025/0.1/transformer/'
tensorboard_dir = ${model_dir}/runs/

mkdir -p $model_dir $tensorboard_dir
torchrun --standalone --nnodes=1 --nproc_per_node=8 vocos/main.py -- \
        --config vocos/configs/default.py \
        --config.train_data=${train_data} \
        --config.model_dir=${model_dir} \
        --config.tensorboard_dir=${tensorboard_dir} \
        --config.max_train_steps 1000000
```

TODO:
- [ ] training 
  - [x] training works
  - [ ] check training process
   - [x] generator
   - [ ] disc
  - [ ] transformer discriminators
- [ ] dev benchmark etc
- [ ] infer
   - [x] offline
   - [ ] chunk by chunk or frame by frame

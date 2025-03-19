import os

base_dir='./data/training/'
tar_dir='./data/training_img/'
os.makedirs(tar_dir)
files=os.listdir('./data/training/')
for file in files:
    pth=os.path.join(base_dir,file)
    tpth=os.path.join(tar_dir,file)
    with open(pth,'r') as F:
        paths = [line.strip() for line in F.readlines() if len(line.strip()) > 0]
        paths=paths[:2935]
    with open(tpth,'w') as f:
        for line in paths:
            f.write(line+'\n')

# import torch
# # export CUDA_VISIBLE_DEVICES=0
# torch.cuda.empty_cache()
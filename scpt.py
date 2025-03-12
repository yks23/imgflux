# import os

# base_dir='/data115/video-diff/workspace/HOI-DiffusionAsShader/data/dexycb_filelist/training/'
# tar_dir='/data115/video-diff/workspace/HOI-DiffusionAsShader/data/dexycb_filelist/training_img/'
# files=os.listdir('/data115/video-diff/workspace/HOI-DiffusionAsShader/data/dexycb_filelist/training/')
# for file in files:
#     pth=os.path.join(base_dir,file)
#     tpth=os.path.join(tar_dir,file)
#     with open(pth,'r') as F:
#         paths = [line.strip() for line in F.readlines() if len(line.strip()) > 0]
#         paths=paths[:2935]
#         with open(tpth,'w') as f:
#             for line in paths:
#                 f.write(line+'\n')

import torch
# export CUDA_VISIBLE_DEVICES=0
torch.cuda.empty_cache()
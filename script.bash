# 直接对比gt 和 infer1000
python -m pytorch_fid ./valid/gtstats.npz ./valid/infer1000 --device cuda:0
# 将gt 和 infer1000 的统计数据保存到文件中
python -m pytorch_fid --save-stats ./valid/gt ./valid/gtstats --device cuda:0
# 使用保存的统计数据进行对比
python -m pytorch_fid ./valid/gtstats.npz ./valid/infer1000 --device cuda:0
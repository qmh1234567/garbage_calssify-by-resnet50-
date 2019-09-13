#  train
python newrun.py --mode='train' --data_local='./../data_set/train_data_v2' --test_data_local='./test_data' --train_local='./output/' --num_classes=40 --max_epochs=5

# test
python newrun.py --mode='test' --data_local='./../data_set/train_data_v2' --test_data_local='./test_data'  --train_local='./output/' --num_classes=40 --max_epochs=5


# # new train
# python newrun.py --mode='train' --train_data_local='./garbage_classify/splitDataset/train' --val_data_local='./garbage_classify/splitDataset/val' --test_data_local='./garbage_classify/test_data' --train_local='./output/' --num_classes=40 --max_epochs=5

# # new test
# python newrun.py --mode='test' --train_data_local='./garbage_classify/splitDataset/train' --val_data_local='./garbage_classify/splitDataset/val' --test_data_local='./garbage_classify/test_data' --train_local='./output/' --num_classes=40 --max_epochs=5


# python newrun_fusion.py --mode='train' --data_local='./garbage_classify/train_data_v2' --test_data_local='./garbage_classify/test_data/' --train_local='./output/' --num_classes=40 --max_epochs=5



# python run.py --data_url='../grabage_code/garbage_classify/train_data_v2' --train_url=' ' --deploy_script_path='./deploy_scripts'

# Dynamic Graph CNN for Learning on Point Clouds (PyTorch)

## Point Cloud Classification
* Run the training script:


``` 1024 points
python main.py --exp_name=dgcnn_1024 --model=dgcnn --num_points=1024 --k=20 --use_sgd=True
```

``` 2048 points
python main.py --exp_name=dgcnn_2048 --model=dgcnn --num_points=2048 --k=40 --use_sgd=True
```

* Run the evaluation script after training finished:

``` 1024 points
python main.py --exp_name=dgcnn_1024_eval --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=checkpoints/dgcnn_1024/models/model.t7
```

``` 2048 points
python main.py --exp_name=dgcnn_2048_eval --model=dgcnn --num_points=2048 --k=40 --use_sgd=True --eval=True --model_path=checkpoints/dgcnn_2048/models/model.t7
```

* Run the evaluation script with pretrained models:

``` 1024 points
python main.py --exp_name=dgcnn_1024_eval --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=pretrained/model.1024.t7
```

``` 2048 points
python main.py --exp_name=dgcnn_2048_eval --model=dgcnn --num_points=2048 --k=40 --use_sgd=True --eval=True --model_path=pretrained/model.2048.t7
```

## PointNet微调和训练

* 全局对比+分组对比损失: `/data/cjj/projects/ContrastLearning/experiment/11-17_15:44_Fusion360GroupContrast/epoch137.pth`

* 仅全局对比: `/data/cjj/projects/ContrastLearning/experiment/11-14_15:32_Fusion360/epoch196.pth`

## DGCNN微调

* 仅全局对比: `/data/cjj/projects/PointCloudLearning/dgcnn/experiment/12-09_20:26_DGCNNFusion360BatchContrast/epoch200.pth`

* 全局对比+分组对比损失: `/data/cjj/projects/PointCloudLearning/dgcnn/experiment/12-08_20:43_Fusion360BatchGroupContrastContinue/epoch200.pth`

## GADNet微调和训练

* 仅全局对比: `/data/cjj/projects/PointCloudLearning/dgcnn/experiment/12-09_23:10_GDANetFusion360BatchContrast/epoch171.pth`

* 
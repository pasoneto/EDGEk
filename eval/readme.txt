#Experiment 1
  - commit for running: Commit 71e1a0b (EDGE redo branch)
  - run python3 create_dataset.py --position_out to create dataset
  - description: Accelerometer data, 30 fps, 10 seconds. full feature set, no foot slide loss, original loss weights, predicting forward kinematics of dance (no angles)
  - weight associated: experiment1.pt
  - predict position from accelerometer

#Experiment 2
  - commit for running f1fc6a7 (EDGE redo branch)
  - run python3 create_dataset.py
  - python3.9 train.py --batch_size 128 --epochs 20000 --save_interval 100 --run_foot_loss (add foot loss in arguments)
  - weight associated: experiment2.pt
  - description: predict angles from accelerometer. foot slide loss, 30fps, 10 seconds. Full feature set. Results worse than exp1.

#Experiment 3
  - commit for running (EDGE redo branch)
  - run python3 create_dataset.py --position_out
  - python3.9 train.py --batch_size 128 --epochs 20000 --save_interval 100
  - weight asociated: experiment3.pt
  - description: predict position from accelerometer, but with smaller window (5 seconds) and smaller fps: 15.

#Experiment 4
  - commit for running (EDGE redo branch)
  - run python3 create_dataset.py --position_out
  - python3.9 train.py --batch_size 128 --epochs 20000 --save_interval 100
  - weight asociated: experiment4.pt
  - description: predict position from accelerometer, but with a bit longer window (10 seconds) and smaller fps: 15.




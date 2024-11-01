#Experiment 1
  - commit for running: Commit 71e1a0b (EDGE redo branch)
  - run python3 create_dataset.py --position_out to create dataset
  - description: Accelerometer data, 30 fps, full feature set, no foot slide loss, original loss weights, predicting forward kinematics of dance (no angles)
  - weight associated: experiment1.pt

#Experiment 2
  - commit for running f1fc6a7 (EDGE redo branch)
  - run python3 create_dataset.py
  - python3.9 train.py --batch_size 128 --epochs 20000 --save_interval 100 --run_foot_loss (add foot loss in arguments)
  - weight associated: experiment2.pt

 



# Learning DL

It fuckin works!  MLP learning with backpropagation written from memory in order to
1. Confirm deep understanding of concept
2. Upgrade intuition
3. Ensure understanding of future chapters of text

Alpha value is unoptimzied- learns much faster at 0.2 with 60 steps

Output looks like this:
```
********************************************************************************
DATASET
********************************************************************************
streetlights: [[0 1 1]
 [1 1 0]
 [0 0 0]
 [1 1 1]]
walk_no_walk [[1]
 [1]
 [0]
 [1]]
********************************************************************************
PARAMETERS
********************************************************************************
hidden_neurons: 4
alpha: 0.1
(3, 4)
weights_0_1: [[-0.16595599  0.44064899 -0.99977125 -0.39533485]
 [-0.70648822 -0.81532281 -0.62747958 -0.30887855]
 [-0.20646505  0.07763347 -0.16161097  0.370439  ]]
weights_1_2: [[-0.5910955 ]
 [ 0.75623487]
 [-0.94522481]
 [ 0.34093502]]
********************************************************************************
INFERENCE TEST
********************************************************************************
layer_1: [[0 1 1]]
layer_2 BEFORE RELU: [[-0.91295327 -0.73768934 -0.78909055  0.06156045]]
layer_2: [[-0.         -0.         -0.          0.06156045]]
layer_3: [[0.02098811]]
delta: [[-0.97901189]]
error: [[0.95846427]]
********************************************************************************
TRAINING
********************************************************************************
ERROR: [[0.05266646]]
ERROR: [[0.05658831]]
ERROR: [[0.05273677]]
ERROR: [[0.04387794]]
ERROR: [[0.03508267]]
ERROR: [[0.02741769]]
ERROR: [[0.02101783]]
ERROR: [[0.01580002]]
ERROR: [[0.0116368]]
ERROR: [[0.008392]]
ERROR: [[0.00592649]]
ERROR: [[0.0041018]]
ERROR: [[0.00278604]]
ERROR: [[0.00186038]]
ERROR: [[0.0012237]]
ERROR: [[0.00079449]]
ERROR: [[0.00051015]]
ERROR: [[0.00032455]]
ERROR: [[0.00020491]]
ERROR: [[0.00012856]]
ERROR: [[8.0257312e-05]]
ERROR: [[4.98985655e-05]]
ERROR: [[3.09229952e-05]]
ERROR: [[1.91140605e-05]]
ERROR: [[1.17905637e-05]]
ERROR: [[7.26127035e-06]]
ERROR: [[4.46617323e-06]]
ERROR: [[2.74423547e-06]]
ERROR: [[1.68485835e-06]]
ERROR: [[1.03379718e-06]]
ERROR: [[6.34008846e-07]]
ERROR: [[3.88677163e-07]]
ERROR: [[2.38205832e-07]]
ERROR: [[1.45953198e-07]]
ERROR: [[8.94117988e-08]]
ERROR: [[5.47662983e-08]]
ERROR: [[3.35415308e-08]]
ERROR: [[2.05406395e-08]]
ERROR: [[1.25780981e-08]]
ERROR: [[7.70180392e-09]]
ERROR: [[4.71575812e-09]]
ERROR: [[2.88732832e-09]]
ERROR: [[1.76778538e-09]]
ERROR: [[1.08231601e-09]]
ERROR: [[6.62630949e-10]]
ERROR: [[4.05680326e-10]]
ERROR: [[2.48365888e-10]]
ERROR: [[1.52053576e-10]]
ERROR: [[9.30890807e-11]]
ERROR: [[5.69900211e-11]]
ERROR: [[3.48897059e-11]]
ERROR: [[2.1359671e-11]]
ERROR: [[1.30764796e-11]]
ERROR: [[8.00546128e-12]]
ERROR: [[4.90096147e-12]]
ERROR: [[3.00037646e-12]]
ERROR: [[1.83683374e-12]]
ERROR: [[1.12451088e-12]]
ERROR: [[6.88425981e-13]]
ERROR: [[4.21454474e-13]]
```

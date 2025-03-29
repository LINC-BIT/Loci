**IFCA** comes from "An Efficient Framework for Clustered Federated Learning". It  proposes a new framework dubbed the Iterative Federated Clustering Algorithm (IFCA), which alternately estimates the cluster identities of the users and optimizes model parameters for the user clusters via gradient descent. The paper shows that in both settings, with good initialization, IFCA converges at an exponential rate, and discuss the optimality of the statistical error rate. When the clustering structure is ambiguous, it proposes to train the models by combining IFCA with the weight sharing technique in multi-task learning. In the experiments, we show that our algorithm can succeed even if we relax the requirements on initialization with random initialization and multiple restarts



We set six continual learning methods on IFCA. You can use the following command to run IFCA:



~~~sh
```shell
cd baselines
cd IFCA  # method name
python mainIFCA_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

python mainIFCA_MAS.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy MAS

python mainIFCA_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM

python mainIFCA_FedKNOW.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy FedKNOW

python mainIFCA_Packnet.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy Packet

python mainIFCA_ChannelGate.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy ChannelGate
~~~


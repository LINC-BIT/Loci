**FedKEMF** comes from "Resource-aware Federated Learning using Knowledge Extraction and Multi-model Fusion". It purposes a resource-aware FL to aggregate an ensemble of local knowledge extracted from edge models, instead of aggregating the weights of each local model, which is then distilled into a robust global knowledge as the server model through knowledge distillation. The local model and the global knowledge are extracted into a tiny size knowledge network by deep mutual learning. Such knowledge extraction allows the edge client to deploy a resourceaware model and perform multi-model knowledge fusion while maintaining communication efficiency and model heterogeneity. Empirical results show that our approach has significantly improved over existing FL algorithms in terms of communication cost and generalization performance in heterogeneous data and models.



We set six continual learning methods on FedKEMF. You can use the following command to run FedKEMF:



~~~sh
```shell
cd baselines
cd FedKEMF  # method name
python mainKEMF_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

python mainKEMF_MAS.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy MAS

python mainKEMF_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM

python mainKEMF_FedKNOW.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy FedKNOW

python mainKEMF_Packnet.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy Packet

python mainKEMF_ChannelGate.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy ChannelGate
~~~


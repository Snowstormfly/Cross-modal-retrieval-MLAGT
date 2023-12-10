# Zero-shot sketch-based remote sensing image retrieval based on multi-level and attention-guided tokenization
https://img.shields.io/badge/python-3.7-green https://img.shields.io/badge/pytorch-1.11-green
The repository is for the paper“Zero-shot sketch-based remote sensing image retrieval based on multi-level and attention-guided tokenization”. In this repository, you can find the official PyTorch implementation of multi-level and attention-guided tokenization network
![网络架构图（新）](https://github.com/Snowstormfly/Cross-modal-retrieval-SAETM/assets/92164018/bd73d19b-34d2-4898-9392-a6e38a1a0ceb)

# Dataset
We provides access to download the RSketch_Ext dataset from [Baidu web disk](https://pan.baidu.com/s/1ieAlTxqkKljcN0EJEk_w2A)
You are free to divide the training set and the test set as you wish

# RSketch_Ext
![数据集示例新](https://github.com/Snowstormfly/Cross-modal-retrieval-SAETM/assets/92164018/dc79aa0e-0fcd-487e-bcea-9d5569826526)

# Train
# pretrained ViT backbone
The pre-trained ViT model on ImageNet-1K is provided on [Baidu Web disk](https://pan.baidu.com/s/19065VR64vuScpRbKQdbuHA) 
You should place sam_ViT-B_16.pth in ./model and modify line 195 in ./model/self_attention.py to absolute path if necessary.


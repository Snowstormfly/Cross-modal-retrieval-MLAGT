# Zero-shot sketch-based remote sensing image retrieval based on multi-level and attention-guided tokenization
 <img src="https://img.shields.io/badge/python-3.7-green"> <img src="https://img.shields.io/badge/pytorch-1.11-green">
 
The repository is for the paper“Zero-shot sketch-based remote sensing image retrieval based on multi-level and attention-guided tokenization”. In this repository, you can find the official PyTorch implementation of multi-level and attention-guided tokenization network
![网络架构图（新）](https://github.com/Snowstormfly/Cross-modal-retrieval-MLAGT/assets/92164018/038d2960-8dc1-4a45-bb07-68956e4219e0)
<h2>Requirements</h2>
<pre>Python 3.7
pytorch 1.11.0
torchvision 0.12.0
einops  0.6.1
</pre>
<h2>Dataset</h2>

We provides access to download the RSketch_Ext dataset from [Baidu web disk](https://pan.baidu.com/s/1ieAlTxqkKljcN0EJEk_w2A)
You are free to divide the training set and the test set as you wish.  (Access Password：vnlc)
<h3>RSketch_Ext</h2>
![RSketch_Ext_dataset](https://github.com/Snowstormfly/Cross-modal-retrieval-MLAGT/assets/RSketch_Ext_dataset.png)

# Train
# pretrained ViT backbone
The pre-trained ViT model on ImageNet-1K is provided on [Baidu Web disk](https://pan.baidu.com/s/19065VR64vuScpRbKQdbuHA) 
You should place sam_ViT-B_16.pth in ./model and modify line 195 in ./model/self_attention.py to absolute path if necessary.


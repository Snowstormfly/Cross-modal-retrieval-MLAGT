# Zero-shot sketch-based remote sensing image retrieval based on multi-level and attention-guided tokenization
 <img src="https://img.shields.io/badge/python-3.7-green"> <img src="https://img.shields.io/badge/pytorch-1.11-green">
 
The repository is for the paper“Zero-shot sketch-based remote sensing image retrieval based on multi-level and attention-guided tokenization”. In this repository, you can find the official PyTorch implementation of multi-level and attention-guided tokenization network
![Network](https://github.com/Snowstormfly/Cross-modal-retrieval-MLAGT/assets/92164018/71876b52-61f5-4cb0-b8ed-fbf1a4e3e30f)
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

![RSketch_Ext_dataset](https://github.com/Snowstormfly/Cross-modal-retrieval-MLAGT/assets/92164018/13693513-6ce5-41d7-bc1a-ce74508debd8)
<h2>Train</h2>
<h3>Pretrained ViT backbone</h3>

The pre-trained ViT model on ImageNet-1K is provided on [Baidu Web disk](https://pan.baidu.com/s/19065VR64vuScpRbKQdbuHA)
You should place sam_ViT-B_16.pth in .\model and modify line 195 in .\model\self_attention.py to absolute path if necessary.  (Access Password：t6p1)
<h3>Argument</h3>
<pre>
# dataset
  train_path           # path to load train data.
  test_path            # path to load test data.
# model
  d_model              # feature dimension.
  d_ff                 # fead-forward layer dimension.
  head                 # number of cross_attention encoder head.
  number               # number of cross_attention encoder layer.
  pretrained           # whether to use pretrained ViT model.
# train
  save                 # model save path.
  batch                # batch size.
  epoch                # train epoch.
  datasetLen           # the amount of data training in a single batch.
  learning_rate        # learning rate.
  weight_decay         # weight_decay.
# test
  load                 # model load path.
  test_sk              # testset number of incoming sketches in a single batch.
  test_im              # testset number of incoming remote sensing image in a single batch.
  num_workers          # dataloader num workers.
  database_path        # preinfer remote sensing image database load path.
  amount               # visualize the number of remote sensing images returned.
  result_path          # accuracy evaluation result saving path.
# other
  choose_cuda          # cuda to use.
  seed                 # random seed.
</pre>

<h2>Conclusion</h2>
Thank you and sorry for the bugs!
<h2>Author</h2>
* Bo Yang <br>
* Chen Wang <br>
* Xiaoshuang Ma <br>
* Beiping Song <br>
* Zhuang Liu
<h2>Citation</h2>
If you think this work is interesting, please cite:
<pre>
 @Article{Cross-modal-retrieval-MLAGT,
    title={Zero-shot sketch-based remote sensing image retrieval based on multi-level and attention-guided tokenization},
    author={Bo Yang, Chen Wang, Xiaoshuang Ma, Beiping Song and Zhuang Liu},
    year={2023},
    journal={},
    volume={},
    number={},
    pages={},
    doi={}
}
</pre>

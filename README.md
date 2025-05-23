<h1>Zero-shot sketch-based remote sensing image retrieval based on multi-level and attention-guided tokenization</h1>

 <img src="https://img.shields.io/badge/python-3.7-green"> <img src="https://img.shields.io/badge/pytorch-1.11-green">
 
The repository is for the paper“Zero-Shot Sketch-Based Remote-Sensing Image Retrieval Based on Multi-Level and Attention-Guided Tokenization”. In this repository, you can find the official PyTorch implementation of multi-level and attention-guided tokenization network
![Network](https://github.com/Snowstormfly/Cross-modal-retrieval-MLAGT/assets/92164018/71876b52-61f5-4cb0-b8ed-fbf1a4e3e30f)
<h2>Requirements</h2>
<pre>Python 3.7
pytorch 1.11.0
torchvision 0.12.0
einops  0.6.1
</pre>
<h2>Dataset</h2>

We provides access to download the RSketch_Ext dataset from [Baidu web disk](https://pan.baidu.com/s/135H35hwRQLiQO7k7fFVBng?pwd=xpmv)
You are free to divide the training set and the test set as you wish.  (Access Password：xpmv)
<h3>RSketch_Ext</h2>

![数据集示例新](https://github.com/Snowstormfly/Cross-modal-retrieval-MLAGT/assets/92164018/99504cfd-85eb-4a7e-8b19-9c7e789e89c3)

<h2>Train and Test</h2>
<h3>Pretrained ViT backbone</h3>

The pre-trained ViT model on ImageNet-1K is provided on [Baidu Web disk](https://pan.baidu.com/s/19065VR64vuScpRbKQdbuHA)
You should place <code>sam_ViT-B_16.pth</code> in <code>./model</code> and modify line 195 in <code>./model/self_attention.py</code> to absolute path if necessary.  (Access Password：t6p1)
<h3>Arguments</h3>
<pre>
# dataset
  train_path              # path to load train data.
  test_path               # path to load test data.
# model
  d_model                 # feature dimension.
  d_ff                    # fead-forward layer dimension.
  head                    # number of cross_attention encoder head.
  number                  # number of cross_attention encoder layer.
  pretrained              # whether to use pretrained ViT model.
# train
  save                    # model save path.
  batch                   # batch size.
  epoch                   # train epoch.
  datasetLen              # the amount of data training in a single batch.
  learning_rate           # learning rate.
  weight_decay            # weight_decay.
# test
  load                    # model load path.
  test_sk                 # testset number of incoming sketches in a single batch.
  test_im                 # testset number of incoming remote sensing image in a single batch.
  num_workers             # dataloader num workers.
  database_path           # preinfer remote sensing image database load path.
  amount                  # visualize the number of remote sensing images returned.
  result_path             # accuracy evaluation result saving path.
# other
  choose_cuda             # cuda to use.
  seed                    # random seed.
</pre>

<h2>Conclusion</h2>
Thank you and sorry for the bugs!
<h2>Author</h2>
* Bo Yang <br>
* Chen Wang <br>
* Xiaoshuang Ma <br>
* Beiping Song <br>
* Zhuang Liu <br>
* Fangde Sun
<h2>Citation</h2>
If you think this work is interesting, please cite:
<pre>
 @Article{Cross-modal-retrieval-MLAGT,
    title={Zero-shot sketch-based remote sensing image retrieval based on multi-level and attention-guided tokenization},
    author={Bo Yang, Chen Wang, Xiaoshuang Ma, Beiping Song, Zhuang Liu and Fangde Sun},
    year={2024},
    journal={Remote Sensing},
    volume={16},
    number={10},
    pages={1653},
    doi={https://doi.org/10.3390/rs16101653}
  }
</pre>

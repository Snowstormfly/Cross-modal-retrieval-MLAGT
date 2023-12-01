from argument import Option
import os
import time
import torch
from torch.utils.data import DataLoader
from utils.util import build_optimizer, save_checkpoint, setup_seed
from utils.datasetbuilding import load_data
from utils.loss import triplet_loss
from model.model import Model
from utils.vaild import valid_cls


def train():
    train_data, sk_valid_data, im_valid_data = load_data(args)

    model = Model(args)
    model = model.cuda()
    # model.to(device)

    optimizer = build_optimizer(args, model)
    train_data_loader = DataLoader(train_data, args.batch, num_workers=2, drop_last=True)
    start_epoch = 0
    accuracy = 0
    for i in range(start_epoch, args.epoch):
        print('------------------------train------------------------')
        epoch = i + 1
        model.train()
        torch.set_grad_enabled(True)
        start_time = time.time()
        num_total_steps = args.datasetLen // args.batch
        for index, (sk, im, sk_neg, im_neg, sk_label, im_label, _, _) in enumerate(train_data_loader):
            # 准备数据
            sk = torch.cat((sk, sk_neg))
            im = torch.cat((im, im_neg))
            sk, im = sk.cuda(), im.cuda()
            # sk, im = sk.to(device), im.to(device)
            cls_fea = model(sk, im)

            # 损失函数
            loss = triplet_loss(cls_fea, args) * 2

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 记录
            step = index + 1
            if step % 30 == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                print(f'epoch_{epoch} step_{step} eta {remaining_time}: loss:{loss.item():.3f} ')
        if epoch >= 10:
            print('------------------------valid------------------------')
            map_all, precision_10, precision_50, precision_100, all_sk_path, image_path_sort = valid_cls(args,
                                                                                                         model,
                                                                                                         sk_valid_data,
                                                                                                         im_valid_data)
            print(
                f'map_all:{map_all:.4f} precision_10:{precision_10:.4f} precision_50:{precision_50:.4f} precision_100:{precision_100:.4f}')
            # 模型保存
            if map_all > accuracy:
                accuracy = map_all
                precision = precision_100
                print("Save the BEST {}th model......".format(epoch))
                save_checkpoint(
                    {'model': model.state_dict(), 'epoch': epoch, 'map_all': accuracy,
                     'precision_100': precision},
                    args.save, f'best_checkpoint_loss_S1')


if __name__ == '__main__':
    args = Option().parse()
    print("train args:", str(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    print("current cuda: " + args.choose_cuda)
    setup_seed(args.seed)

    train()



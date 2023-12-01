import numpy as np
import multiprocessing
from joblib import delayed, Parallel
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


def calculate(distance, class_same, test=None):
    arg_sort_sim = distance.argsort()   # 得到从小到大索引值
    sort_label = []
    for index in range(0, arg_sort_sim.shape[0]):
        # 将label重新排序，根据距离的远近，距离越近的排在前面
        sort_label.append(class_same[index, arg_sort_sim[index, :]])
    sort_label = np.array(sort_label)
    # print(arg_sort_sim, sort_label)

    # 多进程计算
    num_cores = min(multiprocessing.cpu_count(), 4)

    if test:
        start = time.time()

        aps_all = Parallel(n_jobs=num_cores)(
            delayed(voc_eval)(sort_label[iq]) for iq in range(distance.shape[0]))
        map_all = np.nanmean(aps_all)

        precision_10 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 10) for iq in range(sort_label.shape[0]))
        precision_10 = np.nanmean(precision_10)
        precision_50 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 50) for iq in range(sort_label.shape[0]))
        precision_50 = np.nanmean(precision_50)
        precision_100 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 100) for iq in range(sort_label.shape[0]))
        precision_100 = np.nanmean(precision_100)

        print("eval time:", time.time() - start)
        return map_all, precision_10, precision_50, precision_100


def file_name_sort(distance, class_same, file_path, amount: int):
    arg_sort_sim = distance.argsort()  # 得到从小到大索引值
    sort_label = []
    sort_path = []
    for index in range(0, arg_sort_sim.shape[0]):
        # 将label重新排序，根据距离的远近，距离越近的排在前面
        sort_label.append(class_same[index, arg_sort_sim[index, :]])
        sort_path.append(file_path[index, arg_sort_sim[index, :]])
    sort_path = np.array(sort_path)
    sort_path_index = sort_path[:, :amount]  # 返回前a个检索结果的索引值
    return sort_path_index


def voc_eval(sort_class_same, top=None):
    tp = sort_class_same
    tot_pos = np.sum(tp)
    fp = np.logical_not(tp)
    tot = tp.shape[0]
    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        precision = tp / (tp + fp)
    except:
        print("error", tot_pos)
        return np.nan

    ap = voc_ap(rec, precision)
    return ap


def precision_eval(sort_class_same, top=None):
    tp = sort_class_same
    tot_pos = np.sum(tp)

    if top is not None:
        top = min(top, tot_pos)
    else:
        top = tot_pos

    return np.mean(sort_class_same[:top])


def voc_ap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap


def valid_cls(args, model, sk_valid_data, im_valid_data):

    model.eval()
    torch.set_grad_enabled(False)

    print('loading image data')
    sk_dataload = DataLoader(sk_valid_data, batch_size=args.test_sk, num_workers=args.num_workers, drop_last=False)
    print('loading sketch data')
    im_dataload = DataLoader(im_valid_data, batch_size=args.test_im, num_workers=args.num_workers, drop_last=False)

    if args.database_path:
        sk_all = None
        sk_label_all = None
        sk_path_all = None
        for i, (sk, sk_label, sk_path) in enumerate(tqdm(sk_dataload)):
            if i == 0:
                all_sk_label = sk_label.numpy()
                all_sk_path = np.array(sk_path)
            else:
                all_sk_label = np.concatenate((all_sk_label, sk_label.numpy()), axis=0)
                all_sk_path = np.concatenate((all_sk_path, np.array(sk_path)), axis=0)

            sk = sk.cuda()
            sk, sk_inds = model(sk, None, 'test')
            if i == 0:
                sk_all = sk
                sk_label_all = all_sk_label
                sk_path_all = all_sk_path
            else:
                sk_all = torch.cat((sk_all, sk), dim=0)
                sk_label_all = np.concatenate((sk_label_all, sk_label.numpy()), axis=0)
                sk_path_all = np.concatenate((sk_path_all, np.array(sk_path)), axis=0)
        sk_rt = sk_all[:, 0]
        sk_len = sk_rt.size(0)

        data = np.load(args.database_path)
        im_rt = torch.tensor(data['im_rt'])
        im_label_all = data['im_label_all']
        im_path_all = data['im_path_all']
        im_shape = im_rt.size(0),
        im_len = im_shape[0]
        sk_temp = sk_rt.unsqueeze(1).repeat(1, im_len, 1).flatten(0, 1).cuda()
        im_temp = im_rt.unsqueeze(0).repeat(sk_len, 1, 1).flatten(0, 1).cuda()
        feature_1 = torch.cat((sk_temp, im_temp), dim=0)

        dist_temp = F.pairwise_distance(F.normalize(feature_1[:sk_len * im_len]),
                                        F.normalize(feature_1[sk_len * im_len:]), 2)
        dist_im = dist_temp.view(sk_len, im_len).cpu().data.numpy()
        all_dist = dist_im

        # 检索结果可视化
        image_path = np.stack((im_path_all,) * len(sk_label_all), axis=0)
        class_same = (np.expand_dims(sk_label_all, axis=1) == np.expand_dims(im_label_all, axis=0)) * 1
        image_path_sort = file_name_sort(all_dist, class_same, image_path, args.amount)

        # 精度计算
        map_all, precision10, precision50, precision100 = calculate(all_dist, class_same, test=True)

        return map_all, precision10, precision50, precision100, sk_path_all, image_path_sort

    # 按照batch依次读取遥感图像，计算距离后再连接（不爆显存）
    else:
        dist_im = None
        all_dist = None
        for i, (sk, sk_label, sk_path) in enumerate(tqdm(sk_dataload)):
            if i == 0:
                all_sk_label = sk_label.numpy()
                all_sk_path = np.array(sk_path)
            else:
                all_sk_label = np.concatenate((all_sk_label, sk_label.numpy()), axis=0)
                all_sk_path = np.concatenate((all_sk_path, np.array(sk_path)), axis=0)

            sk_len = sk.size(0)
            sk = sk.cuda()
            sk, sk_inds = model(sk, None, 'test')
            for j, (im, im_label, im_path) in enumerate(tqdm(im_dataload)):
                if i == 0 and j == 0:
                    all_im_label = im_label.numpy()
                    all_im_path = np.array(im_path)
                elif i == 0 and j > 0:
                    all_im_label = np.concatenate((all_im_label, im_label.numpy()), axis=0)
                    all_im_path = np.concatenate((all_im_path, np.array(im_path)), axis=0)

                im_len = im.size(0)

                im = im.cuda()
                im, im_inds = model(im, None, 'test')
                sk_temp = sk.unsqueeze(1).repeat(1, im_len, 1, 1).flatten(0, 1).cuda()
                im_temp = im.unsqueeze(0).repeat(sk_len, 1, 1, 1).flatten(0, 1).cuda()
                feature_1 = torch.cat((sk_temp[:, 0], im_temp[:, 0]), dim=0)
                # print(feature_1.size())    # [2*sk*im, 768]
                dist_temp = F.pairwise_distance(F.normalize(feature_1[:sk_len * im_len]),
                                                F.normalize(feature_1[sk_len * im_len:]), 2)
                if j == 0:
                    dist_im = dist_temp.view(sk_len, im_len).cpu().data.numpy()
                else:
                    dist_im = np.concatenate((dist_im, dist_temp.view(sk_len, im_len).cpu().data.numpy()), axis=1)
            if i == 0:
                all_dist = dist_im
            else:
                all_dist = np.concatenate((all_dist, dist_im), axis=0)

        # 检索图像名称排序
        image_path = np.stack((all_im_path,) * len(all_sk_label), axis=0)
        class_same = (np.expand_dims(all_sk_label, axis=1) == np.expand_dims(all_im_label, axis=0)) * 1
        image_path_sort = file_name_sort(all_dist, class_same, image_path, args.amount)

        # 精度计算
        map_all, precision10, precision50, precision100 = calculate(all_dist, class_same, test=True)

        return map_all, precision10, precision50, precision100, all_sk_path, image_path_sort

    # # 将所有遥感图像全部读出再比较距离（会爆显存！）
    # else:
    #     sk_all = None
    #     sk_label_all = None
    #     sk_path_all = None
    #     for i, (sk, sk_label, sk_path) in enumerate(tqdm(sk_dataload)):
    #         if i == 0:
    #             all_sk_label = sk_label.numpy()
    #             all_sk_path = np.array(sk_path)
    #         else:
    #             all_sk_label = np.concatenate((all_sk_label, sk_label.numpy()), axis=0)
    #             all_sk_path = np.concatenate((all_sk_path, np.array(sk_path)), axis=0)
    #
    #         sk = sk.cuda()
    #         sk, sk_inds = model(sk, None, 'test')
    #         if i == 0:
    #             sk_all = sk
    #             sk_label_all = all_sk_label
    #             sk_path_all = all_sk_path
    #         else:
    #             sk_all = torch.cat((sk_all, sk), dim=0)
    #             sk_label_all = np.concatenate((sk_label_all, sk_label.numpy()), axis=0)
    #             sk_path_all = np.concatenate((sk_path_all, np.array(sk_path)), axis=0)
    #     sk_rt = sk_all[:, 0]
    #     sk_len = sk_rt.size(0)
    #
    #     im_all = None
    #     im_label_all = None
    #     im_path_all = None
    #     for j, (im, im_label, im_path) in enumerate(tqdm(im_dataload)):
    #         if j == 0:
    #             all_im_label = im_label.numpy()
    #             all_im_path = np.array(im_path)
    #         else:
    #             all_im_label = np.concatenate((all_im_label, im_label.numpy()), axis=0)
    #             all_im_path = np.concatenate((all_im_path, np.array(im_path)), axis=0)
    #
    #         im = im.cuda()
    #         im, im_inds = model(im, None, 'test')
    #         if j == 0:
    #             im_all = im
    #             im_label_all = all_im_label
    #             im_path_all = all_im_path
    #         else:
    #             im_all = torch.cat((im_all, im), dim=0)
    #             im_label_all = np.concatenate((im_label_all, im_label.numpy()), axis=0)
    #             im_path_all = np.concatenate((im_path_all, np.array(im_path)), axis=0)
    #     im_rt = im_all[:, 0]
    #     im_len = im_rt.size(0)
    #     sk_temp = sk_rt.unsqueeze(1).repeat(1, im_len, 1).flatten(0, 1).cuda()
    #     im_temp = im_rt.unsqueeze(0).repeat(sk_len, 1, 1).flatten(0, 1).cuda()
    #     feature_1 = torch.cat((sk_temp, im_temp), dim=0)
    #
    #     dist_temp = F.pairwise_distance(F.normalize(feature_1[:sk_len * im_len]),
    #                                     F.normalize(feature_1[sk_len * im_len:]), 2)
    #     dist_im = dist_temp.view(sk_len, im_len).cpu().data.numpy()
    #     all_dist = dist_im
    #
    #     # 检索结果可视化
    #     image_path = np.stack((im_path_all,) * len(sk_label_all), axis=0)
    #     class_same = (np.expand_dims(sk_label_all, axis=1) == np.expand_dims(im_label_all, axis=0)) * 1
    #     image_path_sort = file_name_sort(all_dist, class_same, image_path, args.amount)
    #
    #     # 精度计算
    #     map_all, precision10, precision50, precision100 = calculate(all_dist, class_same, test=True)
    #
    #     return map_all, precision10, precision50, precision100, sk_path_all, image_path_sort

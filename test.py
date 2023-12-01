import os
from utils.datasetbuilding import load_data_test
from model.model import Model
from argument import Option
from utils.util import load_checkpoint, setup_seed
import torch
from utils.vaild import valid_cls
import datetime


def test():
    start_time = datetime.datetime.now()
    # 准备数据
    sk_valid_data, im_valid_data = load_data_test(args)

    # 准备模型
    model = Model(args)
    # model = Model(args.d_model, args)
    model = model.half()
    checkpoint = load_checkpoint(args.load)
    cur = model.state_dict()
    new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
    cur.update(new)
    model.load_state_dict(cur)

    if len(args.choose_cuda) > 1:
        model = torch.nn.parallel.DataParallel(model.to('cuda'))
    model = model.cuda()

    # 验证
    map_all, precision_10, precision_50, precision_100, all_sk_path, image_path_sort = valid_cls(args,
                                                                                                 model,
                                                                                                 sk_valid_data,
                                                                                                 im_valid_data)
    seg = "----------\n"
    # 检索结果可视化
    for n in range(len(all_sk_path)):
        if args.result_path is not None:
            f = open(args.result_path, 'a+')
        print('查询图像为：' + all_sk_path[n])
        if args.result_path is not None:
            f.write(seg)
            f.write('查询图像为：' + str(all_sk_path[n]) + '\n')
            f.write('数据库检索结果为：' + '\n')
        for m in range(len(image_path_sort[n, :])):
            print('数据库检索结果为：' + image_path_sort[n, :][m])
            if args.result_path is not None:
                f.write(str(image_path_sort[n, :][m]) + '\n')  # 将检索结果写入txt文件
    print(
        f'map_all:{map_all:.4f} precision_10:{precision_10:.4f} precision_50:{precision_50:.4f} precision_100:{precision_100:.4f}')
    if args.result_path is not None:
        f.write(seg)
        f.write(
            f'map_all:{map_all:.4f} precision_10:{precision_10:.4f} precision_50:{precision_50:.4f} precision_100:{precision_100:.4f}')
        f.close()
    end_time = datetime.datetime.now()
    print("Finish,程序用时为：" + str(end_time - start_time))


if __name__ == '__main__':
    args = Option().parse()
    print("test args:", str(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    print("current cuda: " + args.choose_cuda)
    setup_seed(args.seed)

    test()

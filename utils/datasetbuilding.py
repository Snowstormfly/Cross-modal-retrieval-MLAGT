import os
import numpy as np
from torch.utils import data
from torchvision.transforms import transforms
import cv2
import torch
from PIL import Image
from argument import Option
from utils.util import setup_seed, load_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model import Model


# args = Option().parse()
def train_label_dict(args):
    # 创建训练集label字典
    train_class = os.listdir(args.train_path + os.sep + 'image')
    train_label = list(i for i in range(len(train_class)))
    train_label_dict = dict(zip(train_class, train_label))
    return train_label_dict


def test_label_dict(args):
    # 创建测试集label字典
    test_class = os.listdir(args.test_path + os.sep + 'query')
    test_label = list(i for i in range(len(test_class)))
    test_label_dict = dict(zip(test_class, test_label))
    return test_label_dict


def load_data(args):
    train_class_label, test_class_label = load_para(args)  # cls:类名
    train_data = TrainSet(args, train_class_label)
    sk_valid_data = ValidSet(args, type_skim='sk', half=False, path=False)
    im_valid_data = ValidSet(args, type_skim='im', half=False, path=False)
    return train_data, sk_valid_data, im_valid_data


def load_para(args):
    train_class = []
    for i in os.listdir(args.train_path+os.sep+'sketch'):
        train_class.append(i)
    train_class_label = np.array(train_class)

    test_class = []
    for j in os.listdir(args.test_path+os.sep+'query'):
        test_class.append(j)
    test_class_label = np.array(test_class)

    return train_class_label, test_class_label


def load_data_test(args):
    sk_valid_data = ValidSet(args, type_skim='sk', half=True, path=False)
    im_valid_data = ValidSet(args, type_skim='im', half=True, path=False)
    return sk_valid_data, im_valid_data


class TrainSet(data.Dataset):
    def __init__(self, args, train_class_label):
        self.args = args
        self.train_class_label = train_class_label
        self.choose_label = []
        self.class_dict = create_dict_texts(train_class_label)
        self.all_train_sketch, self.all_train_sketch_label, self.all_train_sketch_cls_name = \
            get_all_train_file(self.args, 'sketch')
        self.all_train_image, self.all_train_image_label, self.all_train_image_cls_name = \
            get_all_train_file(self.args, 'image')

    def __getitem__(self, index):
        # choose 3 label
        self.choose_label_name = np.random.choice(self.train_class_label, 3, replace=False)
        sk_label = self.class_dict.get(self.choose_label_name[0])
        im_label = self.class_dict.get(self.choose_label_name[0])
        sk_label_neg = self.class_dict.get(self.choose_label_name[0])
        im_label_neg = self.class_dict.get(self.choose_label_name[-1])

        sketch = get_file_iccv(self.all_train_sketch_label, self.choose_label_name[0],
                               self.all_train_sketch_cls_name, 1, self.all_train_sketch)
        image = get_file_iccv(self.all_train_image_label, self.choose_label_name[0],
                              self.all_train_image_cls_name, 1, self.all_train_image)
        sketch_neg = get_file_iccv(self.all_train_sketch_label, self.choose_label_name[0],
                                   self.all_train_sketch_cls_name, 1, self.all_train_sketch)
        image_neg = get_file_iccv(self.all_train_image_label, self.choose_label_name[-1],
                                  self.all_train_image_cls_name, 1, self.all_train_image)

        sketch = preprocess(sketch, 'sk')
        image = preprocess(image)
        sketch_neg = preprocess(sketch_neg, 'sk')
        image_neg = preprocess(image_neg)

        # sketch = new_preprocess(sketch, 'sk')
        # image = new_preprocess(image)
        # sketch_neg = new_preprocess(sketch_neg, 'sk')
        # image_neg = new_preprocess(image_neg)
        return sketch, image, sketch_neg, image_neg, \
               sk_label, im_label, sk_label_neg, im_label_neg

    def __len__(self):
        return len(self.all_train_image)


class ValidSet(data.Dataset):

    def __init__(self, args, type_skim='im', half=False, path=False):
        self.args = args
        self.type_skim = type_skim
        self.half = half
        self.path = path
        if type_skim == "sk":
            self.file_names, self.cls = get_all_test_file(self.args, 'sketch')
        elif type_skim == "im":
            self.file_names, self.cls = get_all_test_file(self.args, 'image')
        else:
            NameError(type_skim + " is not right")

    def __getitem__(self, index):
        label = self.cls[index]  # label为数字
        file_name = self.file_names[index]
        if self.path:
            image = file_name
        else:
            if self.half:
                image = preprocess(file_name, self.type_skim).half()
                # image = new_preprocess(file_name, self.type_skim).half()
            else:
                image = preprocess(file_name, self.type_skim)
                # image = new_preprocess(file_name, self.type_skim)
        return image, label, file_name

    def __len__(self):
        return len(self.file_names)


# 每个label，对应一个数字
def create_dict_texts(texts):
    texts = list(texts)
    dicts = {l: i for i, l in enumerate(texts)}
    return dicts


def get_all_train_file(args, skim):
    if skim != 'sketch' or skim != 'image':
        NameError(skim + ' not implemented!')
    if skim == 'sketch':
        train_path = args.train_path+os.sep+skim
        file_list = []
        labels_list = []
        cname_list = []
        for i in os.listdir(train_path):
            cname_list.append(i)
            for j in os.listdir(os.path.join(train_path,i)):
                label_dict = train_label_dict(args)
                labels_list.append(label_dict[i])
                file_path = train_path+os.sep+i+os.sep+j
                file_list.append(file_path)
        file_ls = np.array(file_list)
        labels = np.array(labels_list)
        cname = np.array(cname_list)

    if skim == 'image':
        train_path = args.train_path + os.sep + skim
        file_list = []
        labels_list = []
        cname_list = []
        for i in os.listdir(train_path):
            cname_list.append(i)
            for j in os.listdir(os.path.join(train_path, i)):
                label_dict = train_label_dict(args)
                labels_list.append(label_dict[i])
                file_path = train_path + os.sep + i + os.sep + j
                file_list.append(file_path)
        file_ls = np.array(file_list)
        labels = np.array(labels_list)
        cname = np.array(cname_list)

    return file_ls, labels, cname


def get_file_iccv(labels, class_name, cname, number, file_ls):
    # 该类的label
    label = np.argwhere(cname == class_name)[0, 0]
    # 该类的所有样本
    ind = np.argwhere(labels == label)
    ind_rand = np.random.randint(1, len(ind), number)
    ind_ori = ind[ind_rand]
    files = file_ls[ind_ori][0][0]
    file_path = files
    return file_path


def get_all_test_file(args, skim):
    if skim != 'sketch' or skim != 'image':
        NameError(skim + ' not implemented!')
    if skim == 'sketch':
        test_path = args.test_path+os.sep+'query'
        file_list = []
        labels_list = []
        for i in os.listdir(test_path):
            for j in os.listdir(os.path.join(test_path, i)):
                label_dict = test_label_dict(args)
                labels_list.append(label_dict[i])
                file_path = test_path + os.sep + i + os.sep + j
                file_list.append(file_path)
        file_names = np.array(file_list)
        file_names_cls = np.array(labels_list)

    if skim == 'image':
        test_path = args.test_path + os.sep + 'database'
        file_list = []
        labels_list = []
        for i in os.listdir(test_path):
            for j in os.listdir(os.path.join(test_path, i)):
                label_dict = test_label_dict(args)
                labels_list.append(label_dict[i])
                file_path = test_path + os.sep + i + os.sep + j
                file_list.append(file_path)
        file_names = np.array(file_list)
        file_names_cls = np.array(labels_list)

    return file_names, file_names_cls


def new_preprocess(image_path, img_type="im"):
    # immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
    # imstd = [0.229, 0.224, 0.225]

    immean = [0.5, 0.5, 0.5]  # RGB channel mean for imagenet
    imstd = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(immean, imstd)
    ])

    if img_type == 'im':
        img = np.load(image_path)
        img = torch.Tensor(np.transpose(img, (2, 0, 1)))
        return img
    else:
        # 对sketch 进行crop，等比例扩大到224
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = remove_white_space_image(img, 10)
        img = resize_image_by_ratio(img, 224)
        img = make_img_square(img)
        return transform(img)


def preprocess(image_path, img_type="im"):
    # immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
    # imstd = [0.229, 0.224, 0.225]

    immean = [0.5, 0.5, 0.5]  # RGB channel mean for imagenet
    imstd = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(immean, imstd)
    ])

    if img_type == 'im':
        return transform(Image.open(image_path).resize((224, 224)).convert('RGB'))
    else:
        # 对sketch 进行crop，等比例扩大到224
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = remove_white_space_image(img, 10)
        img = resize_image_by_ratio(img, 224)
        img = make_img_square(img)

        return transform(img)


def remove_white_space_image(img_np: np.ndarray, padding: int):
    """
    获取白底图片中, 物体的bbox; 此处白底必须是纯白色.
    其中, 白底有两种表示方法, 分别是 1.0 以及 255; 在开始时进行检查并且匹配
    对最大值为255的图片进行操作.
    三通道的图无法直接使用255进行操作, 为了减小计算, 直接将三通道相加, 值为255*3的pix 认为是白底.
    """

    h, w, c = img_np.shape
    img_np_single = np.sum(img_np, axis=2)
    Y, X = np.where(img_np_single <= 300)  # max = 300
    ymin, ymax, xmin, xmax = np.min(Y), np.max(Y), np.min(X), np.max(X)
    img_cropped = img_np[max(0, ymin - padding):min(h, ymax + padding), max(0, xmin - padding):min(w, xmax + padding),
                  :]
    return img_cropped


def make_img_square(img_np: np.ndarray):
    if len(img_np.shape) == 2:
        h, w = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1)) * np.max(img_np)
            white2 = np.ones((h, delta2)) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w)) * np.max(img_np)
            white2 = np.ones((delta2, w)) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img
    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1, c), dtype=img_np.dtype) * np.max(img_np)
            white2 = np.ones((h, delta2, c), dtype=img_np.dtype) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w, c), dtype=img_np.dtype) * np.max(img_np)
            white2 = np.ones((delta2, w, c), dtype=img_np.dtype) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img


def resize_image_by_ratio(img_np: np.ndarray, size: int):
    """
    按照比例resize
    """
    # print(len(img_np.shape))
    if len(img_np.shape) == 2:
        h, w = img_np.shape
    elif len(img_np.shape) == 3:
        h, w, _ = img_np.shape
    else:
        assert 0

    ratio = h / w
    if h > w:
        new_img = cv2.resize(img_np, (int(size / ratio), size,))  # resize is w, h  (fx, fy...)
    else:
        new_img = cv2.resize(img_np, (size, int(size * ratio),))
    # new_img[np.where(new_img < 200)] = 0
    return new_img


def create_database(args, im_valid_data, model, database_path):
    im_dataload = DataLoader(im_valid_data, batch_size=args.test_im, num_workers=args.num_workers, drop_last=False)
    im_all = None
    im_label_all = None
    im_path_all = None
    for i, (im, im_label, im_path) in enumerate(tqdm(im_dataload)):
        im = im.cuda()
        im, im_inds = model(im, None, 'test')
        if i == 0:
            im_all = im.cpu().numpy()
            im_label_all = im_label.numpy()
            im_path_all = np.array(im_path)
        else:
            im_all = np.concatenate((im_all, im.cpu().numpy()), axis=0)
            im_label_all = np.concatenate((im_label_all, im_label.numpy()), axis=0)
            im_path_all = np.concatenate((im_path_all, np.array(im_path)), axis=0)
        # print(im_all.shape)
    im_rt = im_all[:, 0]
    print(im_rt.shape)
    print(im_label_all.shape)
    print(im_path_all.shape)
    np.savez(database_path, im_rt=im_rt, im_label_all=im_label_all, im_path_all=im_path_all)
    print("数据库创建完成")


if __name__ == '__main__':
    args = Option().parse()
    print("train args:", str(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    print("current cuda: " + args.choose_cuda)
    setup_seed(args.seed)

    # 创建预推断遥感图像数据库
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
    model.eval()
    torch.set_grad_enabled(False)
    sk_valid_data, im_valid_data = load_data_test(args)
    create_database(args, im_valid_data, model, args.database_path)

import argparse


class Option:

    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")

        # dataset
        parser.add_argument('--train_path', type=str, default=r"E:\research test\ZSE-SBIR\datasets_ext\RSketch_ext\RSketch_S1_ext\train")
        parser.add_argument('--test_path', type=str, default=r"E:\research test\ZSE-SBIR\database\test")

        # model
        parser.add_argument('--d_model', type=int, default=768)
        parser.add_argument('--d_ff', type=int, default=1024)
        parser.add_argument('--head', type=int, default=12)
        parser.add_argument('--number', type=int, default=1)
        parser.add_argument('--pretrained', default=True, action='store_false')

        # train
        parser.add_argument('--save', '-s', type=str, default=r'E:\research test\ZSE-SBIR\datasets\ceshi')
        parser.add_argument('--batch', type=int, default=10)
        parser.add_argument('--epoch', type=int, default=30)
        parser.add_argument('--datasetLen', type=int, default=4800)
        parser.add_argument('--learning_rate', type=float, default=2e-5)
        parser.add_argument('--weight_decay', type=float, default=1e-2)

        # test
        parser.add_argument('--load', '-l', type=str, default=r"E:\research test\ZSE-SBIR\datasets\ceshi\best_checkpoint_loss_S3.pth")
        parser.add_argument('--test_sk', type=int, default=20)
        parser.add_argument('--test_im', type=int, default=20)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--database_path', type=str, default=r"E:\research test\ZSE-SBIR\database\test\gfdatabase.npz")
        parser.add_argument('--amount', type=int, default=5)
        parser.add_argument('--result_path', type=str, default=None)

        # other
        parser.add_argument('--choose_cuda', '-c', type=str, default='0')
        parser.add_argument("--seed", type=int, default=2023, help="random seed.")

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

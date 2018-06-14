import argparse
import networks


class ArgParser(object):
    def __init__(self):
        self.parser = self._create_parser()

    def parse_args(self, args=None):
        args = self.parser.parse_args(args)
        return args

    @staticmethod
    def _create_parser():
        program_name = 'MobileNet-v2 Training Program'
        desc = 'Program for training MobileNet-v2 with periodic evaluation.'
        parser = argparse.ArgumentParser(program_name, description=desc)

        parser.add_argument(
            '--model_dir',
            default="model_mnist",
            type=str,
            help='''Output directory for checkpoints and summaries.'''
        )
        parser.add_argument(
            '--train_data_filename',
            default=r".\mnist\train-images-idx3-ubyte.gz",
            nargs='+',
            type=str,
            help='''Filepaths of the images to be used for training.'''
        )
        parser.add_argument(
            '--train_label_filename',
            default=r".\mnist\train-labels-idx1-ubyte.gz",
            nargs='+',
            type=str,
            help='''Filepaths of the labels to be used for training.'''
        )
        parser.add_argument(
            '--train_sample_num',
            default=60000,
            type=int,
            help='''Number of iamges used for training.'''
        )
        parser.add_argument(
            '--test_data_filename',
            default=r".\mnist\t10k-images-idx3-ubyte.gz",
            nargs='+',
            type=str,
            help='''Filepaths of the images to be used for evaluation.'''
        )
        parser.add_argument(
            '--test_label_filename',
            default=r".\mnist\t10k-labels-idx1-ubyte.gz",
            nargs='+',
            type=str,
            help='''Filepaths of the labels to be used for evaluation.'''
        )
        parser.add_argument(
            '--test_sample_num',
            default=10000,
            type=int,
            help='''Number of iamges used for evaluation.'''
        )
        parser.add_argument(
            '--num_classes',
            default=10,
            type=int,
            help='''Number of classes (unique labels) in the dataset.
                    Ignored if using CIFAR network version.'''
        )
        parser.add_argument(
            '--batch_size',
            default=20,
            type=int,
        )
        parser.add_argument(
            '--learning_rate', '-l',
            type=float,
            default=0.001,
            help='''Initial learning rate for ADAM optimizer.'''
        )
        parser.add_argument(
            '--max_epoch',
            default=100001,
            type=int
        )
        parser.add_argument(
            '--keep_last_n_checkpoints',
            default=3,
            type=int
        )
        parser.add_argument(
            '--depth_multiplier',
            type=float,
            default=0.25,
        )
        parser.add_argument(
            '--min_depth',
            default=8,
            type=int
        )

        return parser

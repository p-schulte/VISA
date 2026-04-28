from argparse import ArgumentParser

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config_loader import CONFIG

BATCHNORM_MOMENTUM = 0.01

class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.mode = None
        self.save_path = None
        self.model_path = None
        self.data_path = None
        self.datasize = None
        self.ckpt = None
        self.optimizer = None
        self.bce_loss = None
        self.lr = 1e-5
        self.enc_layer = 1
        self.dec_layer = 3
        self.nepoch = 10
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        config = CONFIG.dsg

        parser = ArgumentParser(description='training code')
        parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='sgdet', type=str)
        parser.add_argument('-save_path', default='models/dsg_generator/' + config['data_path'].split("/")[-1] + '/', type=str)
        parser.add_argument('-model_path', default=None, type=str)
        parser.add_argument('-data_path', default='/data/scene_understanding/action_genome/', type=str)
        parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
        parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('-nepoch', help='epoch number', default=10, type=float)
        parser.add_argument('-enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
        parser.add_argument('-dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)
        parser.add_argument('-bce_loss', action='store_true',default=True)
        return parser

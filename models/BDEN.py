import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import  BinarizeConv2d,InputScale,SignumActivation,BinarizeTransposedConv2d

__all__ = ['BDEN']

class BDEN(nn.Module):

    def __init__(self, num_classes=1000):
        super(BDEN, self).__init__()
        self.ratioInfl=16
        self.numOfClasses=num_classes

        self.FrontLayer = nn.Sequential(
            InputScale(),
            BinarizeConv2d(3, int(4*self.ratioInfl), kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(int(4*self.ratioInfl)),
            SignumActivation(),
            BinarizeConv2d(int(4*self.ratioInfl), int(4*self.ratioInfl), kernel_size=3, padding=0,stride=1),
            nn.BatchNorm2d(int(4*self.ratioInfl)),
            SignumActivation(),

            BinarizeConv2d(int(4*self.ratioInfl), int(8*self.ratioInfl), kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(int(8*self.ratioInfl)),
            SignumActivation(),
            BinarizeConv2d(int(8*self.ratioInfl), int(8*self.ratioInfl), kernel_size=3, padding=0,stride=1),
            nn.BatchNorm2d(int(8*self.ratioInfl)),
            SignumActivation(),
            BinarizeConv2d(int(8*self.ratioInfl), int(16*self.ratioInfl), kernel_size=3, padding=0,stride=1),
            nn.BatchNorm2d(int(16*self.ratioInfl)),
            SignumActivation(),

            BinarizeTransposedConv2d(int(16*self.ratioInfl), int(16*self.ratioInfl), kernel_size=3, stride=2),
            nn.BatchNorm2d(int(16*self.ratioInfl)),
            SignumActivation(),
            BinarizeTransposedConv2d(int(16*self.ratioInfl), int(8*self.ratioInfl), kernel_size=3, padding=0,stride=1),
            nn.BatchNorm2d(int(8*self.ratioInfl)),
            SignumActivation(),

            BinarizeTransposedConv2d(int(8*self.ratioInfl), int(8*self.ratioInfl), kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(int(8*self.ratioInfl)),
            SignumActivation(),
            BinarizeConv2d(int(8*self.ratioInfl), int(4*self.ratioInfl), kernel_size=3, padding=0,stride=1),
            nn.BatchNorm2d(int(4*self.ratioInfl)),
            SignumActivation()
        )

        self.TailLayer = nn.Sequential(
            BinarizeConv2d(int(4*self.ratioInfl), self.numOfClasses, kernel_size=3, padding=0,stride=1),
            nn.BatchNorm2d(self.numOfClasses),
            nn.Softmax()
        )

        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-2,
        #        'weight_decay': 5e-4, 'momentum': 0.9},
        #    10: {'lr': 5e-3},
        #    15: {'lr': 1e-3, 'weight_decay': 0},
        #    20: {'lr': 5e-4},
        #    25: {'lr': 1e-4}
        #}
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }

    def forward(self, x):
        x = self.FrontLayer(x)
        #x = x.view(-1, 256 * 6 * 6)
        x = self.TailLayer(x)
        return x


def BDEN(**kwargs):
    num_classes = kwargs.get( 'num_classes', 1000)
    return BDEN(num_classes)

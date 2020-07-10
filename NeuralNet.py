import torch.nn as nn

def output_size(input_size, kernel_size, stride = 1, padding = 0):
    width, height = input_size
    out_width = int((width+2*padding)/stride)
    out_height = int((height+2*padding)/stride)
    return (out_width, out_height)

class Net(nn.Module):
    def __init__(self,size=(256,256),norm_size=5, num_classes = 10):
        super().__init__()
        ### Args:
        # size = width,height of input image
        # norm_size: amout of neughboriing channels used for normalization
        # num_classes: Number of classes

        # Input x (3, width, size)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=12, kernel_size=(7,7),stride=4, padding=3)
        self.norm1 = nn.LocalResponseNorm(norm_size)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = (7,7), padding = 3, stride = 1)
        ### stride 4 quarters the filter map size.
        ### To keep the timecomplexity we use four times the number of filters

        # Input x1 (12, width/4, height/4)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3,3), stride=1, padding=1)
        self.norm2 = nn.LocalResponseNorm(norm_size)
        self.relu2 = nn.ReLU()

        # Input x2 (12, width/4, height/4)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3,3), stride=2, padding=1)
        self.relu3 = nn.ReLU()

        ## Shortcut Input x1 (12, width/4, height/4)
        self.short1 = nn.Conv2d(in_channels = 12, out_channels=24, kernel_size=1, stride = 2)
        # Input x3 (24, width/8, height/8)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3,3), stride=1, padding=1)
        self.relu4 = nn.ReLU()

        # Input x4 (24, width/8, height/8)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3,3), stride=2, padding=1)
        self.relu5 = nn.ReLU()

        ## Shortcut Input x3 (24, width/8, height/8)
        self.short2 = nn.Conv2d(in_channels = 24, out_channels=48, kernel_size=1, stride = 2)
        # Input x5 (48, width/16, height/16)
        self.conv6 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3,3), stride=1, padding=1)
        self.relu6 = nn.ReLU()

        # Input x6 (48, width/16, height/16)
        self.conv7 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), stride=2, padding=1)
        self.relu7 = nn.ReLU()

        ## Shortcut Input x5 (48, width/16, height/16)
        self.short3 = nn.Conv2d(in_channels = 48, out_channels=96, kernel_size=1, stride = 2)
        # Input x7 (96, width/32, height/32)
        self.conv8 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), stride=1, padding=1)
        self.relu8 = nn.ReLU()

        self.globalpool = nn.MaxPool2d(kernel_size=(3,3), padding = 1,stride = 1)

        # Output (192, width/32, height/32)
        # Input (-1, width/32, height/32)
        self.fcn1 = nn.Linear(in_features = int(192*size[0]*size[1]/32**2),out_features = 200)
        self.relu9 = nn.ReLU()
        self.fcn2 = nn.Linear(in_features=200, out_features=200)
        self.relu10 = nn.ReLU()
        self.fcn3 = nn.Linear(in_features = 200, out_features = num_classes)
        self.out = nn.Softmax(dim = 1)

    def forward(self,x):
        N, C, W, H = x.shape
        x1 = self.pool1(self.relu1(self.norm1(self.conv1(x))))

        x2 = self.relu2(self.norm2(self.conv2(x1)))

        x3 = self.relu3(self.conv3(x2))

        x4 = self.relu4(self.conv4(x3+self.short1(x1)))

        x5 = self.relu5(self.conv5(x4))

        x6 = self.relu6(self.conv6(x5+self.short2(x3)))

        x7 = self.relu7(self.conv7(x6))

        x8 = self.globalpool(self.relu8(self.conv8(x7+self.short3(x5))))

        x8 = x8.view(-1,int(192*W*H/(32**2)))

        x_out = self.fcn3(self.relu10(self.fcn2(self.relu9(self.fcn1(x8)))))

        x_out = self.out(x_out)
        return x_out

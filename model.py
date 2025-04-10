import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__( )
        self.relu = nn.ReLU()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.skip_connection = nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv_block(x)
        out += identity
        return self.relu(out)

class MyResnet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.init_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv_1 = self.make_block(32, 32, 3, 1)
        self.conv_2 = self.make_block(32, 64, 2, 2)
        self.conv_3 = self.make_block(64, 64, 3, 1)
        self.conv_4 = self.make_block(64, 128, 3, 1)
        self.conv_5 = self.make_block(128, 256, 3, 1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def make_block(self, in_c, out_c, blocks, stride=1):
        layers=[ResidualBlock(in_channels=in_c, out_channels=out_c, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(in_channels=out_c, out_channels=out_c, stride=1))

        return nn.Sequential(*layers)




    def forward(self, x):
        x = self.init_block(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__=="__main__":
    model = MyResnet()
    test = torch.rand(8, 3, 32, 32)
    output = model(test)
    print(output.shape)
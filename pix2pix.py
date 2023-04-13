"""Implementation of pix2pix."""

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm


class GeneratorEncoderDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.enc = nn.Sequential(
            Downsample(in_channels, 64),
            Downsample(64, 128),
            Downsample(128, 256),
            Downsample(256, 512),
            Downsample(512, 512),
            Downsample(512, 512),
            Downsample(512, 512),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.dec = nn.Sequential(
            UpsampleDropout(512, 512),
            UpsampleDropout(512, 512),
            UpsampleDropout(512, 512),
            Upsample(512, 512),
            Upsample(512, 256),
            Upsample(256, 128),
            Upsample(128, 64),
            nn.ConvTranspose2d(
                64,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=False),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.enc(x)
        decoded = self.dec(encoded)
        return decoded


class GeneratorUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.down1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.down1 = Downsample(in_channels, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 512)
        self.down6 = Downsample(512, 512)
        self.down7 = Downsample(512, 512)
        self.down8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.up1 = UpsampleDropout(512, 512)
        self.up2 = UpsampleDropout(1024, 512)
        self.up3 = UpsampleDropout(1024, 512)
        self.up4 = Upsample(1024, 512)
        self.up5 = Upsample(1024, 256)
        self.up6 = Upsample(512, 128)
        self.up7 = Upsample(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=False),
        )

        self.out = nn.Tanh()

    def forward(self, x):
        # Encoder
        # `f` is the feature vector that results from the encoder.
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        f = self.down8(d7)

        # Decoder
        u1_ = self.up1(f)
        u1 = torch.cat([u1_, d7], dim=1)
        u2_ = self.up2(u1)
        u2 = torch.cat([u2_, d6], dim=1)
        u3_ = self.up3(u2)
        u3 = torch.cat([u3_, d5], dim=1)
        u4_ = self.up4(u3)
        u4 = torch.cat([u4_, d4], dim=1)
        u5_ = self.up5(u4)
        u5 = torch.cat([u5_, d3], dim=1)
        u6_ = self.up6(u5)
        u6 = torch.cat([u6_, d2], dim=1)
        u7_ = self.up7(u6)
        u7 = torch.cat([u7_, d1], dim=1)
        y = self.up8(u7)

        return self.out(y)


class PixelDiscriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        xy_concat = torch.cat([x, y], dim=1)
        return self.dis(xy_concat)


class Patch16Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            Downsample(64, 128, stride=1),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        xy_concat = torch.cat([x, y], dim=1)
        return self.dis(xy_concat)


class Patch70Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            Downsample(64, 128),
            Downsample(128, 256),
            Downsample(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        xy_concat = torch.cat([x, y], dim=1)
        return self.dis(xy_concat)


class Patch286Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            Downsample(64, 128),
            Downsample(128, 256),
            Downsample(256, 512),
            Downsample(512, 512),
            Downsample(512, 512),
            Downsample(512, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        xy_concat = torch.cat([x, y], dim=1)
        return self.dis(xy_concat)


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        padding: int = 1
    ):
        super().__init__()

        self.down = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.down(x)


class UpsampleDropout(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.up(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.up(x)


class UpsampleSkipConnection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5),
            nn.ReLU(inplace=False),
        )

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=0)
        return self.up(x)


GENERATORS = {
    "Encoder-Decoder": GeneratorEncoderDecoder,
    "U-net": GeneratorUNet,
}

DISCRIMINATORS = {
    "PixelGAN": PixelDiscriminator,
    "PatchGAN 16x16": Patch16Discriminator,
    "PatchGAN 70x70": Patch70Discriminator,
    "PatchGAN 286x286": Patch286Discriminator,
}

INPUT_SIZE = 256
NUM_EPOCHS = 200
L1_LAMBDA = 100

if torch.cuda.is_available():
    print("CUDA")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("MPS")
    device = torch.device("mps")
else:
    print("CPU")
    device = torch.device("cpu")


def train(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    generator: str,
    discriminator: str,
):
    # Generator.
    gen = GENERATORS[generator](3, 3)
    gen.to(device)
    gen.train()

    # Discriminator.
    dis = DISCRIMINATORS[discriminator](3)
    dis.to(device)
    dis.train()

    # Loss fuctions. BCE for discriminator and L1 for generator.
    bce_loss = nn.BCELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    # Optimizer.
    g_optimizer = optim.Adam(gen.parameters())
    d_optimizer = optim.Adam(dis.parameters())

    # Get image size so all images have the same size. It is also used in
    # separating the input from the label image, since they are placed
    # horizontally next to each other. We assume the images are square.
    initial_datapoint = next(iter(train_loader))[0]
    img_size = initial_datapoint.shape[2]

    for epoch in range(NUM_EPOCHS):
        d_losses = []
        g_losses = []
        print(f"Epoch {epoch}")
        for xy_concat, _ in tqdm(train_loader):
            # Discriminator training.
            dis.zero_grad()

            # Separate input and label image. Left = input, right = label.
            x = xy_concat[:, :, :, :img_size].to(device)
            y = xy_concat[:, :, :, img_size:].to(device)

            x, y = Variable(x).to(device), Variable(y).to(device)
            y_pred = gen(x)

            # The discriminator should predict all ones for "real" images, i.e.
            # images from the dataset.
            d_real_result = dis(x, y)
            d_real_loss = bce_loss(
                d_real_result,
                Variable(torch.ones(d_real_result.shape)).to(device),
            )

            # The discriminator should predict all zeros for "fake" images,
            # i.e. images generated by the generator.
            d_fake_result = dis(x, y_pred)
            d_fake_loss = bce_loss(
                d_fake_result,
                Variable(torch.zeros(d_fake_result.shape)).to(device),
            )

            # Loss of discriminator is the average of the BCE losses of "real"
            # and "fake" data.
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            d_optimizer.step()

            d_losses.append(d_loss.item())

            # Generator training.
            gen.zero_grad()

            y_pred = gen(x)
            d_result = dis(x, y_pred).squeeze()

            # The training loss of the generator is a combination of how well
            # it was able to fool the discriminator (BCE) and how much it
            # resembles the real image (L1).
            g_loss = bce_loss(
                d_result,
                Variable(torch.ones(d_result.size(), device=device))
            ) + L1_LAMBDA * l1_loss(y_pred, y)
            g_loss.backward()
            g_optimizer.step()

            g_losses.append(g_loss.item())

        print("[%d/%d] - d_loss: %.3f, g_loss: %.3f" % (
            epoch + 1,
            NUM_EPOCHS,
            torch.mean(torch.FloatTensor(d_losses)),
            torch.mean(torch.FloatTensor(g_losses)),
        ))


def load_dataset(
    path: str,
    subfolders: list[str],
    transform,
) -> list[torch.utils.data.DataLoader]:
    """Returns data loaders in the same order as `subfolders` and the width of
    the image."""

    data_loaders = []

    for subfolder in subfolders:
        dataset = datasets.ImageFolder(path, transform)
        class_index = dataset.class_to_idx[subfolder]

        n = 0
        for i in range(len(dataset)):
            if dataset.imgs[n][1] != class_index:
                del dataset.imgs[n]
                n -= 1

            n += 1

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )
        data_loaders.append(data_loader)

    return data_loaders


def main():
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE * 2)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    train_loader, val_loader = load_dataset(
        "./data/maps/",
        ["train", "val"],
        transform,
    )

    train(train_loader, val_loader, "U-net", "PatchGAN 70x70")


if __name__ == "__main__":
    main()

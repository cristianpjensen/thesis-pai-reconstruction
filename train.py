import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from rich.progress import track
from architectures import (
    GENERATORS,
    DISCRIMINATORS,
)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    generator: str,
    discriminator: str,
    l1_lambda: int = 50,
    num_epochs: int = 20,
):
    # Generator.
    generator = GENERATORS[generator](3, 3)
    generator.to(device)
    generator.train()

    # Discriminator.
    discriminator = DISCRIMINATORS[discriminator](3)
    discriminator.to(device)
    discriminator.train()

    # Loss fuctions. BCE for discriminator and L1 for generator.
    bce_loss = nn.BCELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    # Optimizer. Initialized to be the same as the baseline.
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=2e-4,
        betas=(0.5, 0.999),
        eps=1e-07,
    )
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=2e-4,
        betas=(0.5, 0.999),
        eps=1e-07,
    )

    # Get image size so all images have the same size. It is also used in
    # separating the input from the label image, since they are placed
    # horizontally next to each other. We assume the images are square.
    initial_datapoint = next(iter(train_loader))[0]
    img_size = initial_datapoint.shape[2]

    for epoch in range(num_epochs):
        d_losses = []
        g_losses = []

        for xy_concat, _ in track(
            train_loader,
            description=f"[{epoch}/{num_epochs}]",
            transient=True,
        ):
            # Discriminator training.
            discriminator.zero_grad()

            # Separate input and label image. Left = input, right = label.
            x = xy_concat[:, :, :, :img_size].to(device)
            y = xy_concat[:, :, :, img_size:].to(device)

            x, y = Variable(x).to(device), Variable(y).to(device)
            y_pred = generator(x)

            # The discriminator should predict all ones for "real" images, i.e.
            # images from the dataset.
            d_real_result = discriminator(x, y)
            d_real_loss = bce_loss(
                Variable(torch.ones(d_real_result.shape)).to(device),
                d_real_result,
            )

            # The discriminator should predict all zeros for "fake" images,
            # i.e. images generated by the generator.
            d_fake_result = discriminator(x, y_pred)
            d_fake_loss = bce_loss(
                Variable(torch.zeros(d_fake_result.shape)).to(device),
                d_fake_result,
            )

            # Loss of discriminator is the sum of the BCE losses of "real"
            # and "fake" data.
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            d_losses.append(d_loss.item())

            # Generator training.
            generator.zero_grad()

            y_pred = generator(x)
            d_result = discriminator(x, y_pred).squeeze()

            # The training loss of the generator is a combination of how well
            # it was able to fool the discriminator (BCE) and how much it
            # resembles the real image (L1).
            g_loss = bce_loss(
                d_result,
                Variable(torch.ones(d_result.size(), device=device))
            ) + l1_lambda * l1_loss(y_pred, y)
            g_loss.backward()
            g_optimizer.step()

            g_losses.append(g_loss.item())

        print("[%d/%d] - d_loss: %.3f, g_loss: %.3f" % (
            epoch + 1,
            num_epochs,
            torch.mean(torch.FloatTensor(d_losses)),
            torch.mean(torch.FloatTensor(g_losses)),
        ))

import torch
from torchvision.transforms import Compose, Resize, RandomAffine
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import os
from Dataset import MyCifarDataset
from model import MyResnet
from Plt_cm import plot_confusion_matrix
def get_args():
    parser = ArgumentParser(description="Hihi")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--tensorboard", type=str)
    parser.add_argument("--check_point", type=str)
    parser.add_argument("--train_model", type=str)
    arg_values = [
        "--epochs", "40",
        "--batch_size", "128",
        "--tensorboard", "tensorboard",
        "--check_point", "check_point",
        # "--train_model", "/content/gdriver/MyDrive/Train_AI/TestCollabcheck_point/last_model.pt"
    ]
    args = parser.parse_args(arg_values)
    return args
if __name__=="__main__":
    args = get_args()
    if torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")

    train_transform = Compose(transforms=[
        RandomAffine(
            scale=(0.9, 1.1),
            degrees=5,
            translate=(0.05, 0.05),
            shear=0.05
        ),
        Resize(32)
    ])

    test_transform = Resize(32)

    train_data = MyCifarDataset(root="", transform=train_transform)
    test_data = MyCifarDataset(root='', transform=test_transform)

    train_dataloader = DataLoader(
        dataset=train_data,
        shuffle=True,
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=2
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        batch_size=args.batch_size
    )

    model=MyResnet().to(device)
    optimizer=torch.optim.Adam(params=model.parameters(), betas=(0.0, 0.999), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    # scheduler = StepLR(step_size=5, gamma=0.5, optimizer=optimizer)
    score = 0

    if args.train_model:
        path=args.train_model
        cp = torch.load(path)
        model.load_state_dict(cp["model"])
        score = cp["accuracy"]
        optimizer.load_state_dict(cp["optimizer"])
        # scheduler.load_state_dict(cp["scheduler"])
        cur_epoch = cp["cur_epoch"]+1

    else:
        cur_epoch=0

    epochs = args.epochs
    num_iters = len(train_dataloader)
    writer = SummaryWriter(log_dir=args.tensorboard)

    if not os.path.isdir(args.check_point):
        os.mkdir(args.check_point)
    for epoch in range(cur_epoch, epochs):
        model.train()
        process_bar=tqdm(train_dataloader)
        for i, (images, labels) in enumerate(process_bar):
            images=images.to(device)
            labels=labels.to(device)
            output = model(images)
            loss_value = criterion(output, labels)
            writer.add_scalar("Train/Loss", loss_value, i+epoch*num_iters)
            process_bar.set_description("Epoch {}/{}, Iteration {}/{}, Loss value: {:.3f}".format(epoch+1, epochs, i+1, num_iters, loss_value))

            model.zero_grad()
            loss_value.backward()
            optimizer.step()

        all_labels = []
        all_predictions = []
        model.eval()
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            all_labels.extend(labels)
            with torch.no_grad():
                output=model(images)
                all_predictions.extend(torch.argmax(output, dim=1))



        all_labels=[o.item() for o in all_labels]
        all_predictions=[o.item() for o in all_predictions]
        accuracy = accuracy_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        writer.add_scalar("Test/Accuracy", accuracy, epoch)

        print(accuracy)

        # scheduler.step()

        check_point = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cur_epoch": epoch,
            "accuracy": accuracy,
            # "scheduler": scheduler.state_dict()
        }

        cm = confusion_matrix(all_labels, all_predictions)
        plot_confusion_matrix(class_names=["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"], writer=writer, epoch=epoch, cm=cm)

        torch.save(check_point, "{}/last_model.pt".format(args.check_point))

        if accuracy>score:
            print("Best model")
            torch.save(check_point, "{}/best_model.pt".format(args.check_point))
            score=accuracy
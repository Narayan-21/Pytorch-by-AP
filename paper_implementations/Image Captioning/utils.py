import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("paper_implementations/Image Captioning/test_examples/dog.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: Dog in a water-body carrying a stick")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1, dataset.vocab))
    )
    test_img2 = transform(
        Image.open(
            "paper_implementations/Image Captioning/test_examples/boy.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: A Boy eating noodles")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2, dataset.vocab))
    )
    test_img3 = transform(Image.open("paper_implementations/Image Captioning/test_examples/man.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: A man sailing through the river.")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3, dataset.vocab))
    )
    test_img4 = transform(
        Image.open(
            "paper_implementations/Image Captioning/test_examples/shoes.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: A person showing off his shoes")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4, dataset.vocab))
    )
    model.train()


def save_checkpoint(state, filename="my_checkpoint_2.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

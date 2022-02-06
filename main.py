# This is a sample Python script.
import os
import torch
import clip
from PIL import Image
from torchvision.datasets import CIFAR100


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def CLIPbasicrun(dev):
    # Use a breakpoint in the code line below to debug your script.
    device = dev  # Press ⌘F8 to toggle the breakpoint.
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open("dog.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a shiba inu", "a dog"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)


def CLIPzeroshot(dev):
    device = dev
    model, preprocess = clip.load('ViT-B/32', device)

    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

    image, class_id = cifar100[3000]
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # CLIPbasicrun("cpu")
    CLIPzeroshot("cpu")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

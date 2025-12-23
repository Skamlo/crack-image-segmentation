from torchvision import transforms


class CrackMaskTransform():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, image, *args, **kwds):
        return self.transform(image, *args, **kwds)

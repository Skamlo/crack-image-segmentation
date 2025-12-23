from torchvision import transforms


class CrackImageTransform():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image, *args, **kwds):
        return self.transform(image, *args, **kwds)

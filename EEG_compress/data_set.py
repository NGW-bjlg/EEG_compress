import  torch
import  os, glob
import  random, csv
from scipy import io

from    torch.utils.data import Dataset, DataLoader

from    torchvision import transforms



class EEG(Dataset):

    def __init__(self, root, mode):
        super(EEG, self).__init__()
        self.root = root
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        self.images, self.labels = self.load_csv('data_file.csv')

        if mode=='train': # 80%
            self.images = self.images[:int(0.8*len(self.images))]
            self.labels = self.labels[:int(0.8*len(self.labels))]
        if mode=='all': # 80%
            self.images = self.images
            self.labels = self.labels
        elif mode == 'test': # 20% = 80%->100%
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 'pokemon\\mewtwo\\00001.png
                images += glob.glob(os.path.join(self.root, name, '*.mat'))

            print(len(images), images)

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        img, label = self.images[idx], self.labels[idx]

        x = io.loadmat(img)
        data = x['sample']
        data1 = data
        tf = transforms.Compose([
            transforms.ToTensor()
        ])
        img = tf(data1)
        label = torch.tensor(label)

        return img, label

def main():

    db = EEG("data_select", 'train')
    x, y = next(iter(db))
    print('sample:', x.shape, y.shape, y)


if __name__ == '__main__':
    main()


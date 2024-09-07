from utils.libraries import *


normalize = transforms.Normalize(mean=[0,0,0], std=[1,1,1])


class ProcessDataset(Dataset):

    def __init__(self, dataset_path='./new_less_data',  phase ='valid', resize= 128, cropsize=128, channel=3):
        
        assert phase in PHASE_NAMES, 'phase: {}, should be in {}'.format(phase, PHASE_NAMES)

        self.dataset_path = dataset_path
        self.phase = phase
        self.resize = resize
        self.cropsize = cropsize

        
        # load dataset: x: image and y:label and mask:ground_truth
        self.x, self.y, self.mask = self.load_dataset_folder()

        
        # set transforms FOR AUGMENTATION
        self.transform_x = transforms.Compose([transforms.Resize(resize),  
                                               transforms.RandomResizedCrop((resize),scale=(0.5,1.0)),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize
])

        self.transform_mask = transforms.Compose([transforms.Resize(resize),
                                               #   transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  normalize


                                                ])
        


    #Function to obtain the     
    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)
        
        
        #normal CT ==> mask= Black
        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])

        else:
            mask = Image.open(mask).convert('RGB')
            mask = self.transform_mask(mask)

        return x, y, mask

        


    def __len__(self):
        return len(self.x)


    def load_dataset_folder(self):
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.phase)
        gt_dir = os.path.join(self.dataset_path, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        
        for img_type in img_types:
            
            img_type_dir = os.path.join(img_dir,  img_type)     
            if not os.path.isdir(img_type_dir):
                continue

            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) ])

                
                #gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png') for img_fname in img_fname_list]
                gt_fpath_list = sorted([os.path.join(gt_type_dir, d) for d in os.listdir(gt_type_dir) ])

                mask.extend(gt_fpath_list)             
       
        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)


for phase in PHASE_NAMES:
    train_dataset = ProcessDataset(data_path,  phase='train')
    test_dataset = ProcessDataset(data_path,  phase='test')
    


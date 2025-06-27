validation_set_size = 0.20
class HC18(Dataset):
    def __init__(self, train=True, transformX=None, transformY=None):
        self.pixel_file = pd.read_csv(os.path.join('training_set_pixel_size_and_HC.csv'))
        self.transformX = transformX
        self.transformY = transformY
        self.train = train
        self.train_data, self.validation_data = train_test_split(self.pixel_file, test_size=validation_set_size,
                                                                 random_state=5)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.validation_data)

    def __getitem__(self, index):
        if self.train:
            imx_name = os.path.join('training_set', self.train_data.iloc[index, 0])
            imy_name = os.path.join('training_set',
                                    self.train_data.iloc[index, 0].replace('.png', '_Annotation.png'))
            true_hc = self.train_data.iloc[index, 2]  
            pixel_size = self.train_data.iloc[index, 1]  # pixel size
        else:
            imx_name = os.path.join('training_set', self.validation_data.iloc[index, 0])
            imy_name = os.path.join('training_set',
                                    self.validation_data.iloc[index, 0].replace('.png', '_Annotation.png'))
            true_hc = self.validation_data.iloc[index, 2] 
            pixel_size = self.validation_data.iloc[index, 1]  # pixel size
        imx = Image.open(imx_name)
        imy = Image.open(imy_name).convert('L')

        
        if self.train:
            # Random horizontal flipping
            if random.random() > 0.5:
                imx = TF.hflip(imx)
                imy = TF.hflip(imy)

            # Random vertical flipping
            if random.random() > 0.5:
                imx = TF.vflip(imx)
                imy = TF.vflip(imy)

            # Random rotation
            if random.random() > 0.8:
                angle = random.choice([-30, -90, -60, -45 - 15, 0, 15, 30, 45, 60, 90])
                imx = TF.rotate(imx, angle)
                imy = TF.rotate(imy, angle)
           
            if random.random() > 0.5:
                shear_angle = random.uniform(-13, 13)
                imx = TF.affine(imx, angle=0, translate=[0, 0], scale=1.0, shear=shear_angle)
                imy = TF.affine(imy, angle=0, translate=[0, 0], scale=1.0, shear=shear_angle)
         
            if random.random() > 0.5:
                max_dx = imx.size[0] * 0.1 
                max_dy = imx.size[1] * 0.1  
                dx = random.uniform(-max_dx, max_dx)
                dy = random.uniform(-max_dy, max_dy)
                imx = TF.affine(imx, angle=0, translate=[dx, dy], scale=1.0, shear=0)
                imy = TF.affine(imy, angle=0, translate=[dx, dy], scale=1.0, shear=0)

            if random.random() > 0.5:
                imx = imx.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
            
            if random.random() > 0.5:            
                brightness_factor = random.uniform(0.8, 1.4)
                contrast_factor = random.uniform(0.8, 1.3)
                imx = TF.adjust_brightness(imx, brightness_factor)
                imx = TF.adjust_contrast(imx, contrast_factor)
        
        if self.transformX:
            imx = self.transformX(imx)
            imy = self.transformY(imy)

        sample = {'image': imx, 'annotation': imy, 'true_hc': true_hc, 'pixel_size': pixel_size}
        return sample

# with open(os.path.join('train_data.pickle'), 'rb') as f:
#     train_data = pickle.load(f)
# with open(os.path.join('validation_data.pickle'), 'rb') as f:
#     validation_data = pickle.load(f)


# Init transform functions
tx_X = transforms.Compose([transforms.Resize((448, 448)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))])
tx_Y = transforms.Compose([transforms.Resize((448, 448)),
                           transforms.ToTensor(),  ################ no need to normalize the maskk
                           # transforms.Normalize((0.5,), (0.5,))
                          ])
train_data = HC18(train = True, transformX = tx_X, transformY = tx_Y)
validation_data = HC18(train = False, transformX = tx_X, transformY = tx_Y )

# Dataloaders

# num_workers = 2 result in error, we can come back to see why; change it to 0 temporarily
train_loader = DataLoader(dataset = train_data, batch_size = 8, shuffle = True, num_workers = 0)
validation_loader = DataLoader(dataset = validation_data, batch_size = 8, shuffle = True, num_workers = 0)

# The following functions will return numpy array from the transformed tensors which were
# obtained from our train_loader. Plot them and see if they are intact
def im_converterX(tensor):
    image = tensor.cpu().clone().detach().numpy() # make copy of tensor and converting it to numpy
                                                # as we will need original later
    image = image.transpose(1,2,0) # swapping axes making (1, 28, 28) image to a (28, 28, 1)
    # print(image.shape)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) # unnormalizing the image
                                                # this also outputs (28, 28, 3) which seems important for plt.imshow
    image = image.clip(0, 1) # to make sure final values are in range 0 to 1 as .ToTensor outputed
    return image

def im_converterY(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1,2,0)
    # print(image.shape)
    image = image * np.array((1, 1, 1))
    image = image.clip(0, 1)
    return image

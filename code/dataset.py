from data_utils import *
import matplotlib.pyplot as plt


class TrainSetLoader(Dataset):
    def __init__(self, num_files, dataset_dir, scale_factor, inType='y'):
        super(TrainSetLoader).__init__()
        self.scale_factor = scale_factor
        self.dir = dataset_dir
        self.num_files = num_files
        if num_files == 1:
            self.train_list = [f"calendar/{i:02d}" for i in range(1, 32)]  # 获取所有文件名
        elif num_files == 2:
            self.train_list = [f"city/{i:02d}" for i in range(1, 32)]  # 获取所有文件名
        elif num_files == 3:
            self.train_list = [f"foliage/{i:02d}" for i in range(1, 32)]  # 获取所有文件名
        elif num_files == 4:
            self.train_list = [f"walk/{i:02d}" for i in range(1, 32)]  # 获取所有文件名

        # with open(dataset_dir+'/sep_trainlist.txt', 'r') as f:
        #     self.train_list = f.read().splitlines()
        self.tranform = augumentation()
        self.inType = inType

    def __getitem__(self, idx):
        HR = []
        LR = []
        for i in range(7):
            if self.num_files == 1:
                img_hr = Image.open(self.dir + f'/calendar/hr/hr_{idx + i + 1:02d}.png')
                img_lr = Image.open(self.dir + f'/calendar/lr_x4/lr_{idx + i + 1:02d}.png')
            elif self.num_files == 2:
                img_hr = Image.open(self.dir + f'/city/hr/hr_{idx + i + 1:02d}.png')
                img_lr = Image.open(self.dir + f'/city/lr_x4/lr_{idx + i + 1:02d}.png')
            elif self.num_files == 3:
                img_hr = Image.open(self.dir + f'/foliage/hr/hr_{idx + i + 1:02d}.png')
                img_lr = Image.open(self.dir + f'/foliage/lr_x4/lr_{idx + i + 1:02d}.png')
            elif self.num_files == 4:
                img_hr = Image.open(self.dir + f'/walk/hr/hr_{idx + i + 1:02d}.png')
                img_lr = Image.open(self.dir + f'/walk/lr_x4/lr_{idx + i + 1:02d}.png')
            # if idx <= 24:
            # elif (idx > 24) and (idx <= 49):
            #     img_hr = Image.open(self.dir + f'/city/hr/hr_{idx + i + 1 -25:02d}.png')
            #     img_lr = Image.open(self.dir + f'/city/lr_x4/lr_{idx + i + 1 -25:02d}.png')
            # elif (idx > 49) and (idx <= 74):
            #     img_hr = Image.open(self.dir + f'/foliage/hr/hr_{idx + i + 1 - 50:02d}.png')
            #     img_lr = Image.open(self.dir + f'/foliage/lr_x4/lr_{idx + i + 1 - 50:02d}.png')
            # else:
            #     img_hr = Image.open(self.dir + f'/walk/hr/hr_{idx + i + 1 - 75:02d}.png')
            #     img_lr = Image.open(self.dir + f'/walk/lr_x4/lr_{idx + i + 1 - 75:02d}.png')
            img_hr = np.array(img_hr, dtype=np.float32)/255.0    # (576, 720, 3)
            img_lr = np.array(img_lr, dtype=np.float32)/255.0    # (144, 180, 3)

            if self.inType == 'y':
                img_hr = rgb2ycbcr(img_hr, only_y=True)[np.newaxis,:]    # (1, 576, 720)
                img_lr = rgb2ycbcr(img_lr, only_y=True)[np.newaxis,:]    # (1, 144, 180)
            if self.inType == 'RGB':
                img_hr = img_hr.transpose(2,0,1)
                img_lr = img_lr.transpose(2,0,1)
            HR.append(img_hr)
            LR.append(img_lr)

        HR = np.stack(HR, 1)  # (1, 7, 576, 720)
        LR = np.stack(LR, 1)  # (1, 7, 144, 180)

        HR, LR = random_crop(HR, LR, 32, self.scale_factor)   # HR(1, 7, 128, 128), LR(1, 7, 32, 32)
        HR, LR = self.tranform(HR, LR)

        HR = torch.from_numpy(np.ascontiguousarray(HR))
        LR = torch.from_numpy(np.ascontiguousarray(LR))

        return LR, HR

    def __len__(self):
        return len(self.train_list) - 6


class ValidSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor, inType='y'):
        super(TrainSetLoader).__init__()
        self.scale_factor = scale_factor
        self.dir = dataset_dir
        self.train_list = [f"calendar/{i:02d}" for i in range(1, 32)]  # 获取所有文件名
        # with open(dataset_dir+'/sep_testlist.txt', 'r') as f:
        #     self.train_list = f.read().splitlines()
        self.tranform = augumentation()
        self.inType = inType
    def __getitem__(self, idx):
        HR = []
        LR = []
        for i in range(7):
            img_hr = Image.open(self.dir + '/sequences/' + self.train_list[idx] + '/im' + str(i + 1) + '.png')
            img_lr = Image.open(self.dir + '/LR_x4/' + self.train_list[idx] + '/im' + str(i + 1) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32)/255.0
            img_lr = np.array(img_lr, dtype=np.float32)/255.0
            if self.inType == 'y':
                img_hr = rgb2ycbcr(img_hr, only_y=True)[np.newaxis, :]
                img_lr = rgb2ycbcr(img_lr, only_y=True)[np.newaxis, :]
            if self.inType == 'RGB':
                img_hr = img_hr.transpose(2, 0, 1)
                img_lr = img_lr.transpose(2, 0, 1)
            HR.append(img_hr)
            LR.append(img_lr)

        HR = np.stack(HR, 1)
        LR = np.stack(LR, 1)

        HR, LR = random_crop(HR, LR, 32, self.scale_factor)
        HR, LR = self.tranform(HR, LR)

        HR = torch.from_numpy(np.ascontiguousarray(HR))
        LR = torch.from_numpy(np.ascontiguousarray(LR))

        return LR, HR
    def __len__(self):
        return len(self.train_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.upscale_factor = scale_factor
        self.img_list = os.listdir(self.dataset_dir + '/lr_x4')
        self.totensor = transforms.ToTensor()
    def __getitem__(self, idx):
        HR = []
        LR = []
        for idx_frame in range(idx - 3, idx + 4):
            if idx_frame < 0:
                idx_frame = 0
            if idx_frame > len(self.img_list) - 1:
                idx_frame = len(self.img_list) - 1
            img_HR = Image.open(self.dataset_dir + '/hr/hr_' + str(idx_frame+1).rjust(2,'0') + '.png')
            img_LR = Image.open(self.dataset_dir + '/lr_x4/lr_' + str(idx_frame + 1).rjust(2, '0') + '.png')
            img_HR = np.array(img_HR, dtype=np.float32) / 255.0
            if idx_frame == idx:
                h, w, c = img_HR.shape
                SR_buicbic = np.array(img_LR.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
                SR_buicbic = rgb2ycbcr(SR_buicbic, only_y=False).transpose(2, 0, 1)
            img_LR = np.array(img_LR, dtype=np.float32) / 255.0
            img_HR = rgb2ycbcr(img_HR, only_y=True)[np.newaxis,:]
            img_LR = rgb2ycbcr(img_LR, only_y=True)[np.newaxis,:]

            HR.append(img_HR)
            LR.append(img_LR)

        HR = np.stack(HR, 1)
        LR = np.stack(LR, 1)

        C, N, H, W= HR.shape
        H = math.floor(H / self.upscale_factor / 4) * self.upscale_factor * 4
        W = math.floor(W / self.upscale_factor / 4) * self.upscale_factor * 4
        HR = HR[:, :, :H, :W]
        SR_buicbic = SR_buicbic[:, :H, :W]
        LR = LR[:, :, :H // self.upscale_factor, :W // self.upscale_factor]

        HR = torch.from_numpy(np.ascontiguousarray(HR))
        LR = torch.from_numpy(np.ascontiguousarray(LR))
        SR_buicbic = torch.from_numpy(np.ascontiguousarray(SR_buicbic))
        return LR, HR, SR_buicbic

    def __len__(self):
        return len(self.img_list)

class InferLoader(Dataset):

    def __init__(self, dataset_dir, scale_factor):
        super(InferLoader).__init__()
        self.dataset_dir = dataset_dir
        self.upscale_factor = scale_factor
        self.img_list = os.listdir(self.dataset_dir + '/lr_x4')
        self.totensor = transforms.ToTensor()
    def __getitem__(self, idx):
        HR = []
        LR = []
        for idx_frame in range(idx - 3, idx + 4):
            if idx_frame < 0:
                idx_frame = 0
            if idx_frame > len(self.img_list) - 1:
                idx_frame = len(self.img_list) - 1
            img_LR_o = Image.open(self.dataset_dir + '/lr_x4/lr_' + str(idx_frame + 1).rjust(2, '0') + '.png')
            img_LR = np.array(img_LR_o, dtype=np.float32) / 255.0
            if idx_frame == idx:
                h, w, c = img_LR.shape
                SR_buicbic = np.array(img_LR_o.resize((w*self.upscale_factor, h*self.upscale_factor), Image.BICUBIC), dtype=np.float32) / 255.0
                SR_buicbic = rgb2ycbcr(SR_buicbic, only_y=False).transpose(2, 0, 1)
            img_LR = rgb2ycbcr(img_LR, only_y=True)[np.newaxis,:]

            LR.append(img_LR)
        LR = np.stack(LR, 1)


        LR = torch.from_numpy(np.ascontiguousarray(LR))
        SR_buicbic = torch.from_numpy(np.ascontiguousarray(SR_buicbic))
        return LR, SR_buicbic

    def __len__(self):
        return len(self.img_list)

class TestSetLoader_Vimeo(Dataset):
    def __init__(self, dataset_dir, video_name, scale_factor, inType='y'):
        super(TestSetLoader).__init__()
        self.upscale_factor = scale_factor
        self.dir = dataset_dir
        self.video_name = video_name
        self.img_list = os.listdir(self.dir + '/sequences/' + self.video_name)
        self.inType = inType

    def __getitem__(self, idx):
        HR = []
        LR = []
        for idx_frame in range(idx - 3, idx + 4):
            if idx_frame < 0:
                idx_frame = 0
            if idx_frame > len(self.img_list) - 1:
                idx_frame = len(self.img_list) - 1
            img_hr = Image.open(self.dir + '/sequences/' + self.video_name + '/im' + str(idx_frame + 1) + '.png')
            img_lr = Image.open(self.dir + '/LR_x4/' + self.video_name + '/im' + str(idx_frame + 1) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32) / 255.0
            if idx_frame == idx:
                h, w, c = img_hr.shape
                SR_buicbic = np.array(img_lr.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
                SR_buicbic = rgb2ycbcr(SR_buicbic, only_y=False).transpose(2, 0, 1)
            img_lr = np.array(img_lr, dtype=np.float32) / 255.0
            if self.inType == 'y':
                img_hr = rgb2ycbcr(img_hr, only_y=True)[np.newaxis, :]
                img_lr = rgb2ycbcr(img_lr, only_y=True)[np.newaxis, :]
            if self.inType == 'RGB':
                img_hr = img_hr.transpose(2, 0, 1)
                img_lr = img_lr.transpose(2, 0, 1)
            HR.append(img_hr)
            LR.append(img_lr)

        HR = np.stack(HR, 1)
        LR = np.stack(LR, 1)

        C, N, H, W = HR.shape
        H = math.floor(H / self.upscale_factor / 4) * self.upscale_factor * 4
        W = math.floor(W / self.upscale_factor / 4) * self.upscale_factor * 4
        HR = HR[:, :, :H, :W]
        SR_buicbic = SR_buicbic[:, :H, :W]
        LR = LR[:, :, :H // self.upscale_factor, :W // self.upscale_factor]

        HR = torch.from_numpy(np.ascontiguousarray(HR))
        LR = torch.from_numpy(np.ascontiguousarray(LR))
        SR_buicbic = torch.from_numpy(np.ascontiguousarray(SR_buicbic))

        return LR, HR, SR_buicbic

    def __len__(self):
        return len(self.img_list)

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        if random.random()<0.5:
            input = input[:, ::-1, :]
            target = target[:, ::-1, :]
        if random.random()<0.5:
            input = input.transpose(0, 1, 3, 2)#C N H W
            target = target.transpose(0, 1, 3, 2)
        return input, target

#*读取配置文件 并执行配置*#
import os
import yaml
from ImageProcessing.FileProcessing import DataFileProcessing
from ImageProcessing.DataTransformation import ImageTransformation
from TrainRun.DataGet import MyDateset
from torch.utils.data import DataLoader
from TrainRun.TrainTestVal import train_test_val
import torch
import time
import tqdm

class Config():
    def __init__(self,address:str):
        yaml_file = yaml.safe_load(open(address, 'r')) #读取配置文件
        self.DataFileAddress = yaml_file['DataFileAddress']
        self.ClassName = yaml_file['ClassName']
        self.ClassColor = yaml_file['ClassColor']
        self.DateLoader = yaml_file['DateLoader']
        self.DataUpdates = yaml_file['DataUpdates']
        self.SaveFileName = yaml_file['SaveFileName']
        self.Epoch = int(yaml_file['Epoch'])
        self.DataSegmentation = yaml_file['DataSegmentation']
        self.LearningRate = float(yaml_file['LearningRate'])
        self.ImageSize = (yaml_file['ImageSize'][0],yaml_file['ImageSize'][1])
        self.BatchSize = int(yaml_file['BatchSize'])
        self.SaveEpoch = int(yaml_file['SaveEpoch'])
        self.ClassColorDictBGR = {}
        self.ClassColorDictRGB = {}

        #创建RGB和BGR的字典索引
        for i in range(len(self.ClassColor)):
            BGRlist = []
            self.ClassColorDictRGB[i] = self.ClassColor[i]
            R = self.ClassColor[i][0]
            G = self.ClassColor[i][1]
            B = self.ClassColor[i][2]
            BGRlist.append(B)
            BGRlist.append(G)
            BGRlist.append(R)
            self.ClassColorDictBGR[i] = BGRlist

        if self.DataUpdates == True:
            update = DataFileProcessing()
            update.update()
        if self.DateLoader == True:
            image = ImageTransformation(self.ClassName,self.ClassColor,self.DataSegmentation,self.DataFileAddress)
            image.jsontolabel()
        
        self.Image_Data = "Image"
        self.Label_Data = "Label"
        self.Train_Data = "./ImageProcessing/Data/Train"
        self.Test_Data = "./ImageProcessing/Data/Test"
        self.Val_Data = "./ImageProcessing/Data/Val"
        time.sleep(0.5)

    def MainRun(self,Cuda_num,Moudle,loss):
        trainDateste = MyDateset(self.Train_Data,self.Image_Data, self.Label_Data, self.ImageSize)
        train_data_loader = DataLoader(trainDateste, batch_size=self.BatchSize, shuffle=True, num_workers=0)
        train_len = len(train_data_loader)
        print("训练集数量:", len(trainDateste))

        testDateste = MyDateset(self.Test_Data, self.Image_Data, self.Label_Data, self.ImageSize)
        test_data_loader = DataLoader(testDateste, batch_size=self.BatchSize, shuffle=True, num_workers=8)
        test_len = len(test_data_loader)
        print("测试集数量:", len(testDateste))

        valDateste = MyDateset(self.Val_Data, self.Image_Data, self.Label_Data, self.ImageSize)
        val_data_loader = DataLoader(valDateste, batch_size=self.BatchSize, shuffle=True, num_workers=8)
        val_len = len(val_data_loader)
        print("验证集数量:", len(valDateste))

        Optimizer = torch.optim.AdamW(Moudle.parameters(), lr=self.LearningRate,weight_decay=0.01)
        learning = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer,T_max=self.Epoch,eta_min=0 )
        # 设备
        if Cuda_num != None:
            # 打印运行设备
            for i in Cuda_num:
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # 显存总量(GB)
                used_memory = torch.cuda.memory_allocated(i) / (1024 ** 3)  # 已使用显存(GB)
                free_memory = total_memory - used_memory
                for i in Cuda_num:
                    print(
                        "CUDA:{},GPU name:{} GPU Total Memory {:.2f}GB ,GPU Used Memory {:.2f}GB ,GPU Free Memory {:.2f}GB".format
                        (i, torch.cuda.get_device_name(i), total_memory, used_memory, free_memory))
        else:
            print("Runing CPU")

        Training_run_time = train_test_val(len(self.ClassName),self.SaveEpoch)

        max_miou = 0
        n = 0
        if self.SaveFileName != None:
            save_date_file = f"./SaveDate/TextDate/{self.SaveFileName}{n}.txt"

            while os.path.exists(save_date_file):
                n += 1
                save_date_file = f"./SaveDate/TextDate/{self.SaveFileName}{n}.txt"
            save_date = open(save_date_file, mode="a")
            save_date.write(f"{Moudle}\n图片尺寸:{self.ImageSize}\nloss:{loss}\nOptimizer:{Optimizer}\nLearningRate:{self.LearningRate}\n")
        else:
            n = 0
            save_date_file = f"./SaveDate/TextDate/SaveData{n}.txt"

            while os.path.exists(save_date_file):
                n += 1
                save_date_file = f"./SaveDate/TextDate/TrainingDate{n}.txt"
            save_date = open(save_date_file, mode="a")
            save_date.write(
                f"{Moudle}\n图片尺寸:{self.ImageSize}\nloss:{loss}\nOptimizer:{Optimizer}\nLearningRate:{self.LearningRate}\n")


        print("准备就绪,开始运行")
        time.sleep(0.1)
        for i in tqdm.tqdm(range(self.Epoch)):
            # 训练
            train_loss_sum, train_pa_sum, train_mIoU_sum = Training_run_time.train(i, train_data_loader, Cuda_num, Moudle,
                                                                       loss, Optimizer, learning,
                                                                        self.ClassColorDictRGB, look=True)
            # 训练平均值
            mean_train_loss = round(train_loss_sum / train_len, 3)
            mean_train_pa = round(train_pa_sum / train_len, 3)
            mean_train_mIoU = round(train_mIoU_sum / train_len, 3)
            train_save_date = {"train_epoch": i + 1, "loss": mean_train_loss, "pa": mean_train_pa,
                               "mIou": mean_train_mIoU}
            save_date.write(f"{train_save_date}\n")
            # 测试
            test_loss_sum, test_pa_sum, test_mIoU_sum = Training_run_time.test(i, test_data_loader, Cuda_num, Moudle, loss,
                                                                    self.ClassColorDictRGB, look=True)

            mean_test_loss = round(test_loss_sum / test_len, 3)
            mean_test_pa = round(test_pa_sum / test_len, 3)
            mean_test_mIou = round(test_mIoU_sum / test_len, 3)
            test_save_date = {"test_epoch": i + 1, "loss": mean_test_loss, "pa": mean_test_pa, "mIou": mean_test_mIou}
            save_date.write(f"{test_save_date}\n")
            if mean_test_mIou >= max_miou and mean_test_mIou != 1.0:
                max_miou = mean_test_mIou
                torch.save(Moudle, "./SaveDate/MoudleData/MoudleBest.pt")
                save_date.write(f"保存最佳轮{i+1}\n")

        # 
        val_loss_sum, val_pa_sum, val_mIoU_sum = Training_run_time.test(i, val_data_loader, Cuda_num, Moudle, loss,
                                                            self.ClassColorDictRGB, look=True)
        mean_val_loss = round(val_loss_sum / val_len, 3)
        mean_val_pa = round(val_pa_sum / val_len, 3)
        mean_val_mIou = round(val_mIoU_sum / val_len, 3)
        val_save_date = {"val_epoch": i + 1, "loss": mean_val_loss, "pa": mean_val_pa, "mIou": mean_val_mIou}
        save_date.write(f"{val_save_date}\n")
        torch.save(Moudle, r"./SaveDate/MoudleData/MoudleLast.pt")








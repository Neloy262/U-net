import torch.nn as nn
import torch

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.double_conv_1 = self.double_conv(1,64)
        self.double_conv_2 = self.double_conv(64,128)
        self.double_conv_3 = self.double_conv(128,256)
        self.double_conv_4 = self.double_conv(256,512)
        self.double_conv_5 = self.double_conv(512,1024)
        
        self.up_conv_1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.up_conv_2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.up_conv_3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.up_conv_4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.out_conv = nn.Conv2d(64,2,kernel_size=1)

        self.double_conv_6 = self.double_conv(1024,512)
        self.double_conv_7 = self.double_conv(512,256)
        self.double_conv_8 = self.double_conv(256,128)
        self.double_conv_9 = self.double_conv(128,64)


    def forward(self,input):

        # contracting path
        x = self.double_conv_1(input)
        cropped_1 = self.center_crop(x,392)        
        x = self.maxpool(x)

        x = self.double_conv_2(x)
        cropped_2 = self.center_crop(x,200)
        x = self.maxpool(x)

        x = self.double_conv_3(x)
        cropped_3 = self.center_crop(x,104)
        x = self.maxpool(x)

        x = self.double_conv_4(x)
        cropped_4 = self.center_crop(x,56)
        x = self.maxpool(x)

        x = self.double_conv_5(x)
        

        # expansive path
        x = self.up_conv_1(x)
       
        x = torch.concat((cropped_4,x),1)
        x = self.double_conv_6(x)

       
        x = self.up_conv_2(x)
        x = torch.concat((cropped_3,x),1)
        x = self.double_conv_7(x)

        x = self.up_conv_3(x)
        x = torch.concat((cropped_2,x),1)
        x = self.double_conv_8(x)

        x = self.up_conv_4(x)
        x = torch.concat((cropped_1,x),1)
        x = self.double_conv_9(x)

        x = self.out_conv(x)

        return x
        

    def double_conv(self,in_c,out_c):
        conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,kernel_size=3),
            nn.ReLU()
        )
        return conv

    

    def center_crop(self,tensor,target_size):
        tensor_size = tensor.size()[2]

        diff = tensor_size - target_size

        new_tensor = tensor[:,:,diff//2:tensor_size-diff//2,diff//2:tensor_size-diff//2]

        return new_tensor
        


a = Unet()
test = torch.rand((1,1,572,572))
x = a.forward(test)
print(x.size())





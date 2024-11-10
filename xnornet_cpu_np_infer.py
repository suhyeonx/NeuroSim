import torch
import torch.nn as nn
import torch.nn.functional as F
# from utee import xnor_quantizer
import numpy as np

class BinActiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        output=input.sign()
        return output
    
    @classmethod
    def Mean(cls,input):
        return torch.mean(input.abs(),1,keepdim=True)

    @staticmethod
    def backward(ctx,grad_output):
        input, =ctx.saved_tensors
        grad_input=grad_output.clone()
        grad_input[input.ge(1)]=0
        grad_input[input.le(-1)]=0
        return grad_input

binactive=BinActiv.apply

class XConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, logger=None,
                wl_input =8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,
                vari=0,t=0,v=0,detect=0,target=0,debug = 0, cuda=True, name = 'Xconv'):
        super(XConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.logger = logger
        self.wl_weight = wl_weight
        self.wl_input = wl_input
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.cuda = cuda
        self.name = name
        
        
    def forward(self, input):  
        if self.inference == 1:
            input1 = binactive(input)
            #scaling factor K
            A = BinActiv.Mean(input1)  
            k=torch.ones(1,1,self.kernel_size[0],self.kernel_size[0]).mul(1/(self.kernel_size[0]**2)).to(input.device)  # Ensure k is on the same device as x
            K = F.conv2d(A, k, stride=self.stride, padding=self.padding)#K의 size은 (64,1,28,28)
            
            #scaling factor alpha
            n=self.weight[0].nelement()
            alpha=self.weight.norm(1,3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n).expand_as(self.weight)#alpha의 크기는 (128,64,3,3)

            #Binarize Weight
            real_weight=self.weight #weights의 크기는 (128,64,3,3)
            #mean_weights와 centered_weights는 (batchnorm)한 느낌이다
            mean_weights = real_weight.mul(-1).mean(dim=1, keepdim=True).expand_as(self.weight).contiguous()
            centered_weights = real_weight.add(mean_weights) 
            cliped_weights = torch.clamp(centered_weights, -1.0, 1.0) #cliped_weights는 tanh 함수를 의미 
            signed_weights = torch.sign(centered_weights).detach() - cliped_weights.detach() + cliped_weights
            binary_weights = signed_weights

            binary_weights=binary_weights.mul(alpha)
            output = F.conv2d(input, binary_weights, self.bias, self.stride, self.padding, self.dilation, self.groups) #x의 크기는 (64,128,28,28)
            output = output.mul(K)

            #weight = float_quantizer.float_range_quantize(self.weight,self.wl_weight)
            #weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            #input = float_quantizer.float_range_quantize(input,self.wl_input)
            #output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            #output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            output= F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)        

        return output


class XLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False,logger = None,
	             wl_input =8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,
                 vari=0,t=0,v=0,detect=0,target=0, cuda=True, name ='XLinear' ):
        super(XLinear, self).__init__(in_features, out_features, bias)
        self.logger = logger
        self.wl_weight = wl_weight
        self.wl_input = wl_input
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.cuda = cuda
        self.name = name

    def forward(self, input):
        # if self.inference == 1:
        #     weight = float_quantizer.float_range_quantize(self.weight,self.wl_weight)
        #     weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
        #     input = float_quantizer.float_range_quantize(input,self.wl_input)
        #     output= F.linear(input, self.weight, self.bias)
        #     output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        # else:
        output= F.linear(input, self.weight, self.bias)
        return output


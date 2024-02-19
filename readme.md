#Project Directory

```
text-to-image-unet/
├── dataset/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── captions.txt  
├── requirements.txt
├── data_prep.py
├── model.py
├── train.py
├── generate.py
└── app.py 

```

## Errors while Running train.py

```

PS D:\Text-2-Image> python -u "d:\Text-2-Image\train.py"
Length of Image Paths: 11
Length of Captions: 11
C:\Users\wbavi\AppData\Local\Programs\Python\Python310\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\wbavi\AppData\Local\Programs\Python\Python310\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Traceback (most recent call last):
  File "d:\Text-2-Image\train.py", line 69, in <module>
    model = UNet(in_channels=3, out_channels=3, text_input_size=512, text_output_size=256).to(DEVICE)
  File "d:\Text-2-Image\model.py", line 60, in __init__
    sample_processed_text = self.text_module(sample_text_features)      
  File "C:\Users\wbavi\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nn\modules\module.py", line 1511, in _wrapped_call_impl  
    return self._call_impl(*args, **kwargs)
  File "C:\Users\wbavi\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nn\modules\module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "d:\Text-2-Image\model.py", line 41, in forward
    output = self.fc(reshaped_input)
  File "C:\Users\wbavi\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nn\modules\module.py", line 1511, in _wrapped_call_impl  
    return self._call_impl(*args, **kwargs)
  File "C:\Users\wbavi\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nn\modules\module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\Users\wbavi\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nn\modules\linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x60 and 512x256)

```

## After update model.py this error coming

```

PS D:\Text-2-Image> python -u "d:\Text-2-Image\train.py"

Length of Image Paths: 11
Length of Captions: 11
C:\Users\wbavi\AppData\Local\Programs\Python\Python310\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\wbavi\AppData\Local\Programs\Python\Python310\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Traceback (most recent call last):
  File "d:\Text-2-Image\train.py", line 80, in <module> 
    text_features = TextProcessingModule(input_size=60, output_size=256)(caption)
  File "C:\Users\wbavi\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nn\modules\module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\wbavi\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\nn\modules\module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "d:\Text-2-Image\model.py", line 53, in forward  
    processed_text = output.unsqueeze(-1).unsqueeze(-1).expand_as(x7)
RuntimeError: expand(torch.FloatTensor{[4, 256, 1, 1]}, size=[4, 256]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (4)

```

## Write your error here 

## Check model.py very carefully and update the parameters - pull request are welcome..

## Solution required...
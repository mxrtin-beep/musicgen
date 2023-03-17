
import torch

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# (B, W, H, C)
# B: Batches
# W: Width
# H: Height
# C: Color
print(f'Shape: {train_images.shape}')		# (5000, 32, 32, 3); stores 0-255 pixel values.


# Make them go from -1.0 to 1.0
train_images = train_images / 127.5 - 1.0
test_images = test_images / 127.5 - 1.0

train_images = torch.from_numpy(train_images)
test_images = torch.from_numpy(test_images)

image = train_images[0, :, :, 0]

def get_padded_image(img_in, padding):
  in_W, in_H = img_in.shape

  padded_W = in_W + 2*padding
  padded_H = in_H + 2*padding

  padded_in = torch.zeros((padded_W, padded_H))

  for i in range(in_W):
    for j in range(in_H):
      padded_in[i + padding, j + padding] = img_in[i, j]
  

  return padded_in

def Conv2D(img_in, kernel, stride = 1, padding = 0):

  assert(stride >= 1)

  k_W, k_H = kernel.shape
  in_W, in_H = img_in.shape
  
  padded_in = get_padded_image(img_in, padding)
  padded_W, padded_H = padded_in.shape

  print(padded_in[0])
  print(padded_in[2])

  assert((padded_W - k_W) % stride == 0)
  assert((padded_H - k_H) % stride == 0)

  assert(stride <= k_W)
  assert(stride <= k_H)

  out_W = int((padded_W - k_W) / stride + 1)
  out_H = int((padded_H - k_H) / stride + 1)

  assert(out_W > 0)
  assert(out_H > 0)

  img_out = torch.empty((out_W, out_H), dtype=torch.float64)


  ixx = 0
  ixy = 0

  out_x = 0
  out_y = 0

  while(ixy <= (padded_H - k_H)):
    while(ixx <= (padded_W - k_W)):
        chunk = padded_in[ixy:(ixy+k_H), ixx:(ixx+k_W)]
        ixx += stride
        prod = torch.tensordot(chunk, kernel)
        img_out[out_y, out_x] = prod

        out_x += 1

    ixx = 0
    ixy += stride

    out_x = 0
    out_y += 1


  return img_out

kernel = torch.ones((1, 1))
out = Conv2D(image, kernel, stride=1, padding=0)
print(torch.allclose(out, image))
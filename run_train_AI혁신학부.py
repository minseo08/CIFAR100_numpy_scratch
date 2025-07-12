import numpy as np
from collections import OrderedDict
from common.functions import *
from tqdm import tqdm

import pickle
import numpy as np
import sys, os
import json 
import logging 
import datetime 
import shutil 
from scipy.ndimage import rotate


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = xp.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = xp.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype=col.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def shuffle_dataset(x, y_fine, y_coarse):
    permutation = np.random.permutation(x.shape[0])
    x_shuffled = x[permutation,:,:,:]
    y_fine_shuffled = y_fine[permutation]
    y_coarse_shuffled = y_coarse[permutation]

    return x_shuffled, y_fine_shuffled, y_coarse_shuffled


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 
    

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
       
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = xp.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = xp.dot(dout, self.W.T)
        self.dW = xp.dot(self.x.T, dout)
        self.db = xp.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None 
        self.y = None    
        self.t = None    
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: 
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[xp.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            keep_prob = 1.0 - self.dropout_ratio
            self.mask = xp.random.rand(*x.shape) < keep_prob
            return (x * self.mask) / keep_prob
        else:
            return x

    def backward(self, dout):
        keep_prob = 1.0 - self.dropout_ratio
        return (dout * self.mask) / keep_prob
    
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        
        self.momentum = momentum
        
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var  
        
        self.batch_size = None
        self.xc = None
        self.std = None
        self.xn = None 
        
        self.dgamma = None
        self.dbeta = None
        
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        
        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        out = self.__forward(x, train_flg) 

        if len(self.input_shape) == 4:
            N, C, H, W = self.input_shape
            out = out.reshape(N, H, W, C)
            out = out.transpose(0, 3, 1, 2)

        return out

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = xp.zeros(D)
            self.running_var = xp.zeros(D)

        if train_flg:
            mu = x.mean(axis=0) 
            xc = x - mu
            var = xp.mean(xc**2, axis=0)
            std = xp.sqrt(var + 1e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn 
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / (xp.sqrt(self.running_var + 1e-7))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        input_shape_original = self.input_shape
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)

        dx = self.__backward(dout) 

        if len(input_shape_original) == 4:
            N, C, H, W = input_shape_original
            dx = dx.reshape(N, H, W, C)
            dx = dx.transpose(0, 3, 1, 2)

        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = xp.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -xp.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = xp.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        self.x = None   
        self.col = None
        self.col_W = None
        
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad) 
        col_W = self.W.reshape(FN, -1).T
        
        out = xp.dot(col, col_W)
        
        if self.b is not None:
            out += self.b
            
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        if self.b is not None:
            self.db = xp.sum(dout, axis=0)
            
        self.dW = xp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = xp.dot(dout, self.col_W.T)
        
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad) 

        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - self.pool_h) / self.stride)
        out_w = int(1 + (W + 2*self.pad - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        
        pool_size = self.pool_h * self.pool_w
        col = col.reshape(-1, pool_size)

        self.arg_max = xp.argmax(col, axis=1)
        out = xp.max(col, axis=1)
        
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = xp.zeros((dout.size, pool_size))
        
        dmax[xp.arange(self.arg_max.size), self.arg_max] = dout.flatten()
        
        N, C, H, W = self.x.shape
        
        out_h = int(1 + (H + 2*self.pad - self.pool_h) / self.stride)
        out_w = int(1 + (W + 2*self.pad - self.pool_w) / self.stride)

        dcol = dmax.reshape(N, C, out_h, out_w, -1)
        dcol = dcol.transpose(0, 2, 3, 1, 4)
        dcol = dcol.reshape(N * out_h * out_w, -1)
        
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
    
def relu(x):
    return xp.maximum(0, x)

def softmax(x):
    x = x - xp.max(x, axis=-1, keepdims=True) 
    return xp.exp(x) / xp.sum(xp.exp(x), axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -xp.sum(xp.log(y[xp.arange(batch_size), t] + 1e-7)) / batch_size

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = xp.zeros_like(val)
                self.v[key] = xp.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * xp.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            update_term = lr_t * self.m[key] / (xp.sqrt(self.v[key]) + 1e-7)
            params[key] -= update_term
            
            if self.weight_decay > 0:
                params[key] -= self.lr * self.weight_decay * params[key]

class ResidualBlock:
    def __init__(self, all_params, param_keys, stride=1):
        """
        - all_params: 네트워크의 전체 self.params 딕셔너리
        - param_keys: 이 블록이 사용해야 할 파라미터의 키 맵
        """
        self.param_keys = param_keys 

        self.conv1 = Convolution(
            W=all_params[param_keys['conv1_w']],
            b=all_params[param_keys['conv1_b']],
            stride=stride,
            pad=1
        )
        self.bn1 = BatchNormalization(
            gamma=all_params[param_keys['bn1_gamma']],
            beta=all_params[param_keys['bn1_beta']]
        )
        self.relu = Relu()
        self.conv2 = Convolution(
            W=all_params[param_keys['conv2_w']],
            b=all_params[param_keys['conv2_b']],
            stride=1,
            pad=1
        )
        self.bn2 = BatchNormalization(
            gamma=all_params[param_keys['bn2_gamma']],
            beta=all_params[param_keys['bn2_beta']]
        )

        self.shortcut = None
        if 'sc_w' in param_keys:
            self.shortcut = Convolution(
                W=all_params[param_keys['sc_w']],
                b=all_params[param_keys['sc_b']],
                stride=stride,
                pad=0
            )
            self.shortcut_bn = BatchNormalization(
                gamma=all_params[param_keys['sc_bn_gamma']],
                beta=all_params[param_keys['sc_bn_beta']]
            )
        
        self.layers = [self.conv1, self.bn1, self.relu, self.conv2, self.bn2]

    def forward(self, x, train_flg=True):
        identity = x
        out = x
        for layer in self.layers:
            if isinstance(layer, (BatchNormalization, Dropout)):
                out = layer.forward(out, train_flg)
            else:
                out = layer.forward(out)
        
        if self.shortcut:
            identity = self.shortcut.forward(identity)
            identity = self.shortcut_bn.forward(identity, train_flg)
        
        out += identity
        out = self.relu.forward(out) 
        
        return out

    def backward(self, dout):
        dout = self.relu.backward(dout) 
        
        d_identity = dout 
        d_main = dout

        if self.shortcut:
            d_identity = self.shortcut_bn.backward(d_identity)
            d_identity = self.shortcut.backward(d_identity)

        for layer in reversed(self.layers):
            d_main = layer.backward(d_main)
            
        return d_main + d_identity

class ResNet18:
    def __init__(self, input_dim=(3, 32, 32), output_size=100, dropout_ratio=0.3):
                
        self.params = {}
        self.layers = OrderedDict()

        # --- 1. 초기 컨볼루션 및 배치 정규화 계층 ---
        self.params['W1'] = xp.random.randn(64, input_dim[0], 3, 3) * xp.sqrt(2.0 / (input_dim[0] * 9))
        self.params['gamma1'] = xp.ones(64)
        self.params['beta1'] = xp.zeros(64)
        
        self.layers['conv1'] = Convolution(self.params['W1'], None, stride=1, pad=1)
        self.layers['bn1'] = BatchNormalization(gamma=self.params['gamma1'], beta=self.params['beta1'])
        self.layers['relu1'] = Relu()

        # --- 2. Residual Block 계층들 ---
        self.layers['conv2_x'] = self._make_layer(in_channels=64, out_channels=64, num_blocks=2, stride=1, stage_name='2')
        self.layers['conv3_x'] = self._make_layer(in_channels=64, out_channels=128, num_blocks=2, stride=2, stage_name='3')
        self.layers['conv4_x'] = self._make_layer(in_channels=128, out_channels=256, num_blocks=2, stride=2, stage_name='4')
        self.layers['conv5_x'] = self._make_layer(in_channels=256, out_channels=512, num_blocks=2, stride=2, stage_name='5')

        # --- 3. 최종 계층들 ---
        self.layers['avgpool'] = Pooling(pool_h=4, pool_w=4, stride=1, pad=0)
        
        # AvgPool과 Affine 계층 사이에 Dropout 계층을 추가
        self.layers['dropout'] = Dropout(dropout_ratio)
        logging.info(f"Dropout: {dropout_ratio}")
        
        self.params['W_affine'] = xp.random.randn(512, output_size) * xp.sqrt(2.0 / 512)
        self.params['b_affine'] = xp.zeros(output_size)
        self.layers['affine'] = Affine(self.params['W_affine'], self.params['b_affine'])
        
        self.last_layer = SoftmaxWithLoss()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, stage_name):
        
        layers = []
        
        # --- 첫 번째 블록  ---
        param_keys_0 = {
            'conv1_w': f'W_{stage_name}_0_conv1', 'conv1_b': f'b_{stage_name}_0_conv1',
            'bn1_gamma': f'gamma_{stage_name}_0_bn1', 'bn1_beta': f'beta_{stage_name}_0_bn1',
            'conv2_w': f'W_{stage_name}_0_conv2', 'conv2_b': f'b_{stage_name}_0_conv2',
            'bn2_gamma': f'gamma_{stage_name}_0_bn2', 'bn2_beta': f'beta_{stage_name}_0_bn2',
        }
        self.params[param_keys_0['conv1_w']] = xp.random.randn(out_channels, in_channels, 3, 3) * xp.sqrt(2.0 / (in_channels * 9))
        self.params[param_keys_0['conv1_b']] = xp.zeros(out_channels)
        self.params[param_keys_0['bn1_gamma']] = xp.ones(out_channels)
        self.params[param_keys_0['bn1_beta']] = xp.zeros(out_channels)
        self.params[param_keys_0['conv2_w']] = xp.random.randn(out_channels, out_channels, 3, 3) * xp.sqrt(2.0 / (out_channels * 9))
        self.params[param_keys_0['conv2_b']] = xp.zeros(out_channels)
        self.params[param_keys_0['bn2_gamma']] = xp.ones(out_channels)
        self.params[param_keys_0['bn2_beta']] = xp.zeros(out_channels)

        if stride != 1 or in_channels != out_channels:
            param_keys_0.update({
                'sc_w': f'W_{stage_name}_0_shortcut', 'sc_b': f'b_{stage_name}_0_shortcut',
                'sc_bn_gamma': f'gamma_{stage_name}_0_shortcut_bn', 'sc_bn_beta': f'beta_{stage_name}_0_shortcut_bn'
            })
            self.params[param_keys_0['sc_w']] = xp.random.randn(out_channels, in_channels, 1, 1) * xp.sqrt(2.0 / in_channels)
            self.params[param_keys_0['sc_b']] = xp.zeros(out_channels)
            self.params[param_keys_0['sc_bn_gamma']] = xp.ones(out_channels)
            self.params[param_keys_0['sc_bn_beta']] = xp.zeros(out_channels)

        block0 = ResidualBlock(self.params, param_keys_0, stride)
        layers.append(block0)

        # --- 나머지 블록들 ---
        for i in range(1, num_blocks):
            param_keys_i = {
                'conv1_w': f'W_{stage_name}_{i}_conv1', 'conv1_b': f'b_{stage_name}_{i}_conv1',
                'bn1_gamma': f'gamma_{stage_name}_{i}_bn1', 'bn1_beta': f'beta_{stage_name}_{i}_bn1',
                'conv2_w': f'W_{stage_name}_{i}_conv2', 'conv2_b': f'b_{stage_name}_{i}_conv2',
                'bn2_gamma': f'gamma_{stage_name}_{i}_bn2', 'bn2_beta': f'beta_{stage_name}_{i}_bn2',
            }
            self.params[param_keys_i['conv1_w']] = xp.random.randn(out_channels, out_channels, 3, 3) * xp.sqrt(2.0 / (out_channels * 9))
            self.params[param_keys_i['conv1_b']] = xp.zeros(out_channels)
            self.params[param_keys_i['bn1_gamma']] = xp.ones(out_channels)
            self.params[param_keys_i['bn1_beta']] = xp.zeros(out_channels)
            self.params[param_keys_i['conv2_w']] = xp.random.randn(out_channels, out_channels, 3, 3) * xp.sqrt(2.0 / (out_channels * 9))
            self.params[param_keys_i['conv2_b']] = xp.zeros(out_channels)
            self.params[param_keys_i['bn2_gamma']] = xp.ones(out_channels)
            self.params[param_keys_i['bn2_beta']] = xp.zeros(out_channels)
            
            block_i = ResidualBlock(self.params, param_keys_i, stride=1)
            layers.append(block_i)
            
        return layers

    def predict(self, x, train_flg=True):
        for layer in self.layers.values():
            if isinstance(layer, list): 
                for block in layer:
                    x = block.forward(x, train_flg)
            elif isinstance(layer, (BatchNormalization, Dropout)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t_fine, t_coarse, fine_to_coarse_map):
        y = self.predict(x, train_flg=False)
        y_pred_fine = xp.argmax(y, axis=1)
        
        if t_fine.ndim != 1: t_fine = xp.argmax(t_fine, axis=1)
        acc_fine = xp.sum(y_pred_fine == t_fine) / float(x.shape[0])
        
        y_pred_coarse = fine_to_coarse_map[y_pred_fine]
        if t_coarse.ndim != 1: t_coarse = xp.argmax(t_coarse, axis=1)
        acc_coarse = xp.sum(y_pred_coarse == t_coarse) / float(x.shape[0])
        
        return {'fine': acc_fine, 'coarse': acc_coarse}

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)
        dout = 1
        dout = self.last_layer.backward(dout)
        tmp_layers = list(self.layers.values())
        tmp_layers.reverse()
        for layer_or_list in tmp_layers:
            if isinstance(layer_or_list, list):
                for block in reversed(layer_or_list):
                    dout = block.backward(dout)
            else:
                dout = layer_or_list.backward(dout)
        grads = {}
        for stage_name, num_blocks in [('', 1), ('2', 2), ('3', 2), ('4', 2), ('5', 2)]:
            if stage_name == '': # 초기 레이어
                grads['W1'] = self.layers['conv1'].dW
                grads['gamma1'] = self.layers['bn1'].dgamma
                grads['beta1'] = self.layers['bn1'].dbeta
                continue

            for i in range(num_blocks):
                block = self.layers[f'conv{stage_name}_x'][i]
                param_keys = block.param_keys

                grads[param_keys['conv1_w']] = block.conv1.dW
                grads[param_keys['conv1_b']] = block.conv1.db
                grads[param_keys['bn1_gamma']] = block.bn1.dgamma
                grads[param_keys['bn1_beta']] = block.bn1.dbeta
                
                grads[param_keys['conv2_w']] = block.conv2.dW
                grads[param_keys['conv2_b']] = block.conv2.db
                grads[param_keys['bn2_gamma']] = block.bn2.dgamma
                grads[param_keys['bn2_beta']] = block.bn2.dbeta

                if block.shortcut:
                    grads[param_keys['sc_w']] = block.shortcut.dW
                    grads[param_keys['sc_b']] = block.shortcut.db
                    grads[param_keys['sc_bn_gamma']] = block.shortcut_bn.dgamma
                    grads[param_keys['sc_bn_beta']] = block.shortcut_bn.beta

        grads['W_affine'] = self.layers['affine'].dW
        grads['b_affine'] = self.layers['affine'].db
        return grads

    def save_params(self, file_path="resnet18_params.pkl"):
        params_to_save = {key: val for key, val in self.params.items()}

        params_to_save['bn1_running_mean'] = self.layers['bn1'].running_mean
        params_to_save['bn1_running_var'] = self.layers['bn1'].running_var

        for stage_name in ['2', '3', '4', '5']:
            for i, block in enumerate(self.layers[f'conv{stage_name}_x']):
                params_to_save[f'bn_{stage_name}_{i}_1_running_mean'] = block.bn1.running_mean
                params_to_save[f'bn_{stage_name}_{i}_1_running_var'] = block.bn1.running_var
                params_to_save[f'bn_{stage_name}_{i}_2_running_mean'] = block.bn2.running_mean
                params_to_save[f'bn_{stage_name}_{i}_2_running_var'] = block.bn2.running_var
                if block.shortcut:
                    params_to_save[f'bn_{stage_name}_{i}_s_running_mean'] = block.shortcut_bn.running_mean
                    params_to_save[f'bn_{stage_name}_{i}_s_running_var'] = block.shortcut_bn.running_var
        with open(file_path, 'wb') as f:
            pickle.dump(params_to_save, f)

    def load_params(self, file_path="resnet18_params.pkl"):
        if not os.path.exists(file_path):
            raise IOError(f"파라미터 파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        def to_current_device(arr):
            if arr is None: return None
            if xp == np:
                return cp.asnumpy(arr) if 'cupy' in str(type(arr)) else arr
            else:
                return cp.asarray(arr) if 'numpy' in str(type(arr)) else arr

        # 1. self.params 딕셔너리에 있는 모든 파라미터 업데이트
        for key, val in self.params.items():
            if key in loaded_data:
                self.params[key] = to_current_device(loaded_data[key])
        
        # 2. 배치 정규화 통계량(running_mean, running_var)을 불러와 할당
        self.layers['bn1'].running_mean = to_current_device(loaded_data['bn1_running_mean'])
        self.layers['bn1'].running_var = to_current_device(loaded_data['bn1_running_var'])

        for stage_name in ['2', '3', '4', '5']:
            for i, block in enumerate(self.layers[f'conv{stage_name}_x']):
                block.bn1.running_mean = to_current_device(loaded_data.get(f'bn_{stage_name}_{i}_1_running_mean'))
                block.bn1.running_var = to_current_device(loaded_data.get(f'bn_{stage_name}_{i}_1_running_var'))
                block.bn2.running_mean = to_current_device(loaded_data.get(f'bn_{stage_name}_{i}_2_running_mean'))
                block.bn2.running_var = to_current_device(loaded_data.get(f'bn_{stage_name}_{i}_2_running_var'))
                if block.shortcut:
                    block.shortcut_bn.running_mean = to_current_device(loaded_data.get(f'bn_{stage_name}_{i}_s_running_mean'))
                    block.shortcut_bn.running_var = to_current_device(loaded_data.get(f'bn_{stage_name}_{i}_s_running_var'))

        logging.info(f"Parameters loaded from {file_path} and correctly assigned to all layers.")
    
    
def evaluate_dataset(network, x_data, y_fine, y_coarse, fine_to_coarse_map, batch_size, data_name="Data Set"):
    logging.info(f"Starting evaluation on {data_name} with batch processing...")
    num_data = x_data.shape[0]
    total_loss = 0
    total_acc_fine = 0
    total_acc_coarse = 0

    # 배치 루프
    for i in tqdm(range(0, num_data, batch_size), desc=f"Evaluating {data_name}"):
        x_batch = x_data[i:i+batch_size]
        y_batch_fine = y_fine[i:i+batch_size]
        y_batch_coarse = y_coarse[i:i+batch_size]

        loss = network.loss(x_batch, y_batch_fine, train_flg=False)
        accs = network.accuracy(x_batch, y_batch_fine, y_batch_coarse, fine_to_coarse_map)

        current_batch_size = x_batch.shape[0]
        total_loss += loss.item() * current_batch_size
        total_acc_fine += accs['fine'].item() * current_batch_size
        total_acc_coarse += accs['coarse'].item() * current_batch_size

    # 최종 평균 계산
    final_loss = total_loss / num_data
    final_accs = {
        'fine': total_acc_fine / num_data,
        'coarse': total_acc_coarse / num_data
    }
    
    # 결과를 딕셔너리로 반환
    return {'loss': final_loss, 'accs': final_accs}

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='latin1')
    return dict_data

def load_cifar100_data(path):
    meta = unpickle(os.path.join(path, "meta"))
    fine_label_names = meta['fine_label_names']
    coarse_label_names = meta['coarse_label_names'] 
    
    train_data = unpickle(os.path.join(path, "train"))
    X_train = train_data['data'].astype('float32').reshape(-1, 3, 32, 32) / 255.0
    y_train_fine = xp.array(train_data['fine_labels'])
    y_train_coarse = xp.array(train_data['coarse_labels']) 
    
    test_data = unpickle(os.path.join(path, "test"))
    X_test = test_data['data'].astype('float32').reshape(-1, 3, 32, 32) / 255.0
    y_test_fine = xp.array(test_data['fine_labels'])
    y_test_coarse = xp.array(test_data['coarse_labels']) 
    
    return (X_train, y_train_fine, y_train_coarse), (X_test, y_test_fine, y_test_coarse), fine_label_names, coarse_label_names

def augment_half_flip_half_rotate(X, y, Y, angle=10):
    """
    X: (N, C, H, W), y: (N,)
    -> Return: X_total (100000, C, H, W), y_total (100000,)
    """
    N = X.shape[0]
    
    assert N == 50000, "입력은 CIFAR-100 원본(5만장)이어야 합니다."

    # 무작위 인덱스 섞기 후 반 나누기
    indices = np.random.permutation(N)
    idx_flip = indices[:N//2]   # 25,000
    idx_rotate = indices[N//2:] # 25,000

    # 증강 수행
    flipped = np.flip(X[idx_flip], axis=3)  # 좌우 플립
    rotated = np.zeros_like(X[idx_rotate])
    for i, idx in enumerate(idx_rotate):
        for c in range(3):  # 채널별 회전
            rotated[i, c] = rotate(
                X[idx, c], angle=angle, reshape=False, order=1, mode='constant', cval=0.0
            )

    # 증강된 이미지 및 라벨 결합
    X_aug = np.concatenate([flipped, rotated], axis=0)
    y_aug = np.concatenate([y[idx_flip], y[idx_rotate]], axis=0)
    Y_aug = np.concatenate([Y[idx_flip], Y[idx_rotate]], axis=0)

    # 원본과 증강 결합
    X_total = np.concatenate([X, X_aug], axis=0)
    y_total = np.concatenate([y, y_aug], axis=0)
    Y_total = np.concatenate([Y, Y_aug], axis=0)
    
    return X_total, y_total, Y_total
    
if __name__ == '__main__':
    # ======================================================================
    # 0. cupy 설정
    # ======================================================================
    GPU = True
 
    if GPU:
        import cupy as cp
        xp = cp
        print("GPU(CuPy)를 사용합니다.")
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    else:
        xp = np
    print("CPU(NumPy)를 사용합니다.")
    
    # ======================================================================
    # 1. 설정 및 로깅
    # ======================================================================
    CONFIG_FILE = 'config.json' 
    RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    def load_config(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config

    config = load_config(CONFIG_FILE)

    EXPERIMENT_RUN_DIR = os.path.join(config['output_dir_base'], f"{config.get('experiment_name', 'ResNet_Experiment')}_{RUN_TIMESTAMP}")
    os.makedirs(EXPERIMENT_RUN_DIR, exist_ok=True)

    shutil.copy(CONFIG_FILE, os.path.join(EXPERIMENT_RUN_DIR, 'config.json'))

    LOG_FILE = os.path.join(EXPERIMENT_RUN_DIR, 'experiment_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("===================================================")
    logging.info(f"Starting ResNet Experiment: {config.get('experiment_name', 'ResNet_Experiment')}")
    logging.info(f"Run Timestamp: {RUN_TIMESTAMP}")
    logging.info(f"Results will be saved in: {EXPERIMENT_RUN_DIR}")
    logging.info("===================================================")
    logging.info("Loaded configuration:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
    logging.info("===================================================")



    #  ======================================================================
    # 2. 데이터 로드 및 전처리
    # ======================================================================
    (x_train, y_train_fine, y_train_coarse), (x_test, y_test_fine, y_test_coarse), fine_label_names, coarse_label_names = load_cifar100_data(config['dataset_path'])
    
    print("증강 전:", x_train.shape, y_train_coarse.shape) 
    
    x_train, y_train_fine,y_train_coarse = augment_half_flip_half_rotate(x_train, y_train_fine,y_train_coarse)
    
    print("증강 후:", x_train.shape, y_train_fine.shape) 

    if GPU:
        x_train = xp.asarray(x_train)
        y_train_fine = xp.asarray(y_train_fine)
        y_train_coarse = xp.asarray(y_train_coarse)
        x_test = xp.asarray(x_test)
        y_test_fine = xp.asarray(y_test_fine)
        y_test_coarse = xp.asarray(y_test_coarse)
        logging.info("데이터 로딩 및 GPU 전송 완료.")

    logging.info(f"Train shapes: Data={x_train.shape}, FineLabels={y_train_fine.shape}, CoarseLabels={y_train_coarse.shape}")
    logging.info(f"Test shapes: Data={x_test.shape}, FineLabels={y_test_fine.shape}, CoarseLabels={y_test_coarse.shape}")

    if GPU:
        temp_y_fine_cpu = cp.asnumpy(y_train_fine)
        temp_y_coarse_cpu = cp.asnumpy(y_train_coarse)
    else:
        temp_y_fine_cpu = y_train_fine
        temp_y_coarse_cpu = y_train_coarse
        
    map_cpu = np.zeros(100, dtype=np.int32)
    for fine_idx, coarse_idx in zip(temp_y_fine_cpu, temp_y_coarse_cpu):
        map_cpu[fine_idx] = coarse_idx
    fine_to_coarse_map = xp.asarray(map_cpu)
    logging.info("Created fine-to-coarse label mapping.")

    # ======================================================================
    # 3. 학습/검증 데이터 분리
    # ======================================================================
    validation_rate = config['validation_rate']
    validation_num = int(x_train.shape[0] * validation_rate)

    x_train_shuffled, y_train_fine_shuffled, y_train_coarse_shuffled = shuffle_dataset(x_train, y_train_fine, y_train_coarse)

    x_valid, t_valid_fine, t_valid_coarse = x_train_shuffled[:validation_num], y_train_fine_shuffled[:validation_num], y_train_coarse_shuffled[:validation_num]
    x_train_final, y_train_final_fine, y_train_final_coarse = x_train_shuffled[validation_num:], y_train_fine_shuffled[validation_num:], y_train_coarse_shuffled[validation_num:]

    num_train, num_valid = x_train_final.shape[0], x_valid.shape[0]
    idx_train, idx_valid = xp.arange(num_train), xp.arange(num_valid)

    logging.info(f"Final Training data: {x_train_final.shape}, {y_train_final_fine.shape}")
    logging.info(f"Validation data: {x_valid.shape}, {t_valid_fine.shape}")

    # ======================================================================
    # 4. ResNet 모델 및 옵티마이저 생성
    # ======================================================================
    # --- 하이퍼파라미터 설정 ---
    max_epochs = config['max_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    optimizer_type = config['optimizer_type']
    dropout_rate = config['dropout']
    
    # --- ResNet-18 인스턴스 생성 ---
    network = ResNet18(input_dim=(3, 32, 32), output_size=100, dropout_ratio=dropout_rate)
    logging.info("Network architecture: ResNet-18")

    if optimizer_type.lower() == 'adam':
        optimizer = Adam(lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(lr=learning_rate)
    elif optimizer_type.lower() == 'adamw':
        optimizer = AdamW(lr=learning_rate)
    else:
        logging.warning(f"Optimizer type '{optimizer_type}' not recognized. Using Adam with lr={learning_rate}.")
        optimizer = Adam(lr=learning_rate)

    logging.info(f"Optimizer: {optimizer_type}, Learning Rate: {learning_rate}")

    # ======================================================================
    # 5. 훈련 루프
    # ======================================================================
    train_loss_list, valid_loss_list = [], []
    train_acc_fine_list, train_acc_coarse_list = [], []
    valid_acc_fine_list, valid_acc_coarse_list = [], []


    iter_per_train_epoch = max(num_train // config['batch_size'], 1)
    best_valid_acc = 0.0
    best_model_params_path = os.path.join(EXPERIMENT_RUN_DIR, 'best_resnet18_params.pkl')
    logging.info(f"Best model parameters will be saved to: {best_model_params_path}")

    logging.info("Starting training...")
    for iepoch in tqdm(range(config['max_epochs'])):
        xp.random.shuffle(idx_train)

        for i in tqdm(range(iter_per_train_epoch)):
            batch_mask = idx_train[i*config['batch_size'] : (i+1)*config['batch_size']]
            x_batch, y_batch_fine = x_train_final[batch_mask], y_train_final_fine[batch_mask]

            grads = network.gradient(x_batch, y_batch_fine)
            optimizer.update(network.params, grads)

        train_acc_fine_sum, train_acc_coarse_sum, train_loss_sum = 0, 0, 0
        train_total_num = x_train_final.shape[0]

        for i in range(0, train_total_num, batch_size):
            x_batch = x_train_final[i:i+batch_size]
            y_batch_fine = y_train_final_fine[i:i+batch_size]
            y_batch_coarse = y_train_final_coarse[i:i+batch_size]

            accs = network.accuracy(x_batch, y_batch_fine, y_batch_coarse, fine_to_coarse_map)
            loss = network.loss(x_batch, y_batch_fine, train_flg=False)

            train_acc_fine_sum += accs['fine'].item() * len(x_batch)
            train_acc_coarse_sum += accs['coarse'].item() * len(x_batch)
            train_loss_sum += loss.item() * len(x_batch)

        avg_epoch_train_acc_fine = train_acc_fine_sum / train_total_num
        avg_epoch_train_acc_coarse = train_acc_coarse_sum / train_total_num
        avg_epoch_train_loss = train_loss_sum / train_total_num

        train_acc_fine_list.append(avg_epoch_train_acc_fine)
        train_acc_coarse_list.append(avg_epoch_train_acc_coarse)
        train_loss_list.append(avg_epoch_train_loss)

        if num_valid > 0:
            valid_acc_fine_sum, valid_acc_coarse_sum, valid_loss_sum = 0, 0, 0
            valid_total_num = x_valid.shape[0]

            for i in range(0, valid_total_num, batch_size):
                x_batch = x_valid[i:i+batch_size]
                t_batch_fine = t_valid_fine[i:i+batch_size]
                t_batch_coarse = t_valid_coarse[i:i+batch_size]
                
                accs = network.accuracy(x_batch, t_batch_fine, t_batch_coarse, fine_to_coarse_map)
                loss = network.loss(x_batch, t_batch_fine, train_flg=False)

                valid_acc_fine_sum += accs['fine'].item() * len(x_batch)
                valid_acc_coarse_sum += accs['coarse'].item() * len(x_batch)
                valid_loss_sum += loss.item() * len(x_batch)
                
            avg_epoch_valid_acc_fine = valid_acc_fine_sum / valid_total_num
            avg_epoch_valid_acc_coarse = valid_acc_coarse_sum / valid_total_num
            avg_epoch_valid_loss = valid_loss_sum / valid_total_num

            valid_acc_fine_list.append(avg_epoch_valid_acc_fine)
            valid_acc_coarse_list.append(avg_epoch_valid_acc_coarse)
            valid_loss_list.append(avg_epoch_valid_loss)
            
            # F/S ==> Fine Class/Super Class
            logging.info(f"[Epoch {iepoch+1}] Train Loss: {avg_epoch_train_loss:.4f}, Acc(F/S): {avg_epoch_train_acc_fine:.4f}/{avg_epoch_train_acc_coarse:.4f} | "
                        f"Valid Loss: {avg_epoch_valid_loss:.4f}, Acc(F/S): {avg_epoch_valid_acc_fine:.4f}/{avg_epoch_valid_acc_coarse:.4f}")

            if avg_epoch_valid_acc_coarse > best_valid_acc:
                best_valid_acc = avg_epoch_valid_acc_coarse
                network.save_params(best_model_params_path)
                
                logging.info(f"*** Best validation accuracy (fine) updated: {best_valid_acc:.4f}. Model saved. ***")

    logging.info("Training finished.")

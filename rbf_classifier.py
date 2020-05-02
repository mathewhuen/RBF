import torch
from torch import nn
import torch.functional as F

class RBF(nn.Module):
    """
    A pytorch RBF layer (a more general version of https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py)
    Accepts any input shape.
    
    """
    def __init__(self, d_in, d_out, k, basis_fnc='gaussian', init_args=None, classify=True):#d_in and d_out must be tuples with first dimension corresponding to the squeezed shape of a single input to batch index
        """
        
        Arguments
        ---------
        d_in: int,tuple
            The dimensions of the input data (excluding the batch dimension).
        
        d_out: int
            the number of outputs (classes).
        
        k: int
            The number of RBF kernels to use.
            
        basis_fnc: str, optional
            The type of kernel to use (phi). Default is gaussian.
            
        init_args:  dict, optional
            A custom dictionary defining initialization for paramters. Default in None.
            
            This dictionary must contain keys for kernel 'centers' and any other parameters used in the kernel function. Each key should map to another dictionary with keys 'fnc' and 'args' corresponding to an initialization function (eg, nn.init.normal_) and a kwargs dict for said initialization function, respectively.
        
        classifiy: bool, optional
            Indicates whether or not a dense layer with softmax activation should be appended to the results of the rbf kernel. Default is True
            
            If classify is set to False, d_out is ignored, and the output is of the shape (batch size, k).
        """
        super(RBF, self).__init__()
        self.d_in = d_in if type(d_in)==tuple else (d_in,)
        self.d_out = d_out
        self.k = k
        self.classify = classify
        
        basis_fncs = {
            'gaussian':gaussian,
            'inverse_quadratic':inverse_quadratic
        }
        
        basis_fnc_params = {
            'gaussian' : {
                'gamma':{
                    'fnc':nn.init.constant_,
                    'dims':(self.k,),
                    'args':{'val':1}
                }
            },
            'inverse_quadratic' : {
            }
        }
        
        
        
        self.basis_fnc = basis_fncs[basis_fnc]
        
        params = {}
        params.update([('centers', nn.Parameter(torch.Tensor(1, *self.d_in, k))), 
                       *[(key, nn.Parameter(torch.Tensor(*(basis_fnc_params[basis_fnc][key]['dims'])))) for key in basis_fnc_params[basis_fnc].keys()]])
        self.params = nn.ParameterDict(params)
        
        if(init_args==None):
            init_args = {}
            init_args.update([('centers', {'fnc': nn.init.normal_, 'args':{'mean':0, 'std':1}}), 
                       *[(key, {'fnc':basis_fnc_params[basis_fnc][key]['fnc'], 'args':basis_fnc_params[basis_fnc][key]['args']}) for key in basis_fnc_params[basis_fnc].keys()]])
        self._initialize_params(init_args)
        
        if(self.classify):
            self.net = nn.Sequential()#later set flag for this...I might want just output of rbf layer
            self.net.add_module('rbf_weights', nn.Linear(self.k, self.d_out))
            self.net.add_module('softmax', nn.Softmax(dim=1))#later add a threshold to this

    def _initialize_params(self, kwargs):
        for param in self.params.keys():
            kwargs[param]['fnc'](self.params[param], **(kwargs[param]['args']))
            
    def forward(self, x):
        if(len(x.shape)==1):
            x = x.reshape(1,x.shape[0])
        if(self.d_in!=x[0].shape):
            error_str = 'Actual input dimension ({ain}) does not match specified input dimension ({spin}) for all dim>=1. Please note that RBF does not yet support variable shaped inputs. If you are working with variable shaped convolutional inputs, consider using global max pooling to force the dimension to the number of feature maps or another dimension forcing technique like Spatial Pyramid Pooling.'.format(ain=x.shape, spin=(1, *self.d_in))
            raise TypeError(error_str)
        dim_mod = [1]*(len(self.d_in)+1)
#         print(dim_mod)
#         dist = self.params['centers']-x.reshape(-1, *dim_mod).repeat(*dim_mod,self.k)#use squeeze and unsqueeze and expand
        dist = self.params['centers']-x.reshape(*x.shape, 1).repeat(*dim_mod,self.k)#use squeeze and unsqueeze and expand
#         print(self.params['centers'].shape, x.shape)
        
        r = self.basis_fnc(dist, {key:self.params[key] for key in self.params.keys() if key!='centers'})
        
        y = self.net(r) if self.classify else r
        
        return(y)
    
    
    
def gaussian(dist, params, p=2):
    """
    """
    n = len(dist.shape)
    n_obs = dist.shape[0]
    gamma = params['gamma']
    dist = dist.pow(p).sum(tuple(range(1,n-1))) * gamma.unsqueeze(0).expand(n_obs, -1)
    phi = torch.exp(-dist)
    return(phi)
    
def inverse_quadratic(dist, params, p=2):
    n = len(dist.shape)
    n_obs = dist.shape[0]
    dist = dist.pow(p).sum(tuple(range(1,n-1)))
    phi = torch.ones_like(dist) / (torch.ones_like(dist) + dist)
    return phi

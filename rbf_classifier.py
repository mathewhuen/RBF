class RBF(nn.Module):
    """
    A pytorch RBF layer (a more general version of https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py)
    Accepts any input shape.
    """
    def __init__(self, d_in, d_out, k, basis_fnc='gaussian', init_args=None):#d_in and d_out must be tuples with first dimension corresponding to the squeezed shape of a single input to batch index
        super(RBF, self).__init__()
        basis_fncs = {
            'gaussian':gaussian,
            'inverse_quadratic':inverse_quadratic
        }
        
        basis_fnc_params = {
            'gaussian' : {
                'gamma':{
                    'fnc':nn.init.constant_,
                    'dims':(k,),
                    'args':{'val':1}
                }
            },
            'inverse_quadratic' : {
            }
        }
        
        self.d_in = d_in
        self.d_out = d_out
        self.k = k
        
        self.basis_fnc = basis_fncs[basis_fnc]
        
        params = {}
        params.update([('centers', nn.Parameter(torch.Tensor(1, *d_in, k))), 
                       *[(key, nn.Parameter(torch.Tensor(*(basis_fnc_params[basis_fnc][key]['dims'])))) for key in basis_fnc_params[basis_fnc].keys()]])
        self.params = nn.ParameterDict(params)
        
        if(init_args==None):
            init_args = {}
            init_args.update([('centers', {'fnc': nn.init.normal_, 'args':{'mean':0, 'std':1}}), 
                       *[(key, {'fnc':basis_fnc_params[basis_fnc][key]['fnc'], 'args':basis_fnc_params[basis_fnc][key]['args']}) for key in basis_fnc_params[basis_fnc].keys()]])
        self.initialize_params(init_args)
        
        self.net = nn.Sequential()#later set flag for this...I might want just output of rbf layer
        self.net.add_module('rbf_weights', nn.Linear(k, d_out))
        self.net.add_module('softmax', nn.Softmax(dim=1))#later add a threshold to this
        
    def initialize_params(self, kwargs):
        for param in self.params.keys():
#             print(param)
#             print(self.params[param])
#             print(kwargs[param]['args'])
            kwargs[param]['fnc'](self.params[param], **(kwargs[param]['args']))
            
    def forward(self, x):
        if(self.d_in!=x[0].shape):
            error_str = 'Actual input dimension ({ain}) does not match specified input dimension ({spin}) for all dim>=1. Please note that RBF does not yet support variable shaped inputs. If you are working with variable shaped convolutional inputs, consider using global max pooling to force the dimension to the number of feature maps or another dimension forcing technique like Spatial Pyramid Pooling.'.format(ain=x.shape, spin=(1, *self.d_in))
            raise TypeError(error_str)
        dim_mod = [1]*(len(self.d_in)+1)
        dist = self.params['centers']-x.reshape(-1, *dim_mod).repeat(*dim_mod,self.k)#use squeeze and unsqueeze and expand

        r = self.basis_fnc(dist, {key:self.params[key] for key in self.params.keys() if key!='centers'})
        
        y = self.net(r)
        
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

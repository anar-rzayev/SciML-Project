import torch
import torch.nn as nn

from .diffusion_fn import MultiplicativeNoise, AdditiveNoise
from .integrated_flow import IntegratedFlow
from .flow_fn import RandFlowFn, RandFlowFn_v2
from .flow_net import MultiScaleFlow

from .layers.conv2d import RandConv2d
from .layers.linear import RandLinear
from .layers.groupnorm2d import RandGroupNorm

def rand_norm(dim, **rand_args):
    return RandGroupNorm(min(32, dim), dim, **rand_args)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


def heavy_downsampling(in_nc, nc_hidden, **rand_args):
    downsampling_layers = [
        RandConv2d(in_nc, nc_hidden, 3, 1, 1, **rand_args),
        rand_norm(nc_hidden, **rand_args),
        nn.ReLU(inplace=True),
        RandConv2d(nc_hidden, nc_hidden * 2, 3, 1, 1, **rand_args),
        rand_norm(nc_hidden * 2, **rand_args),
        nn.ReLU(inplace=True),
        RandConv2d(nc_hidden * 2, nc_hidden * 4, 4, 2, 1, **rand_args),
    ]
    return downsampling_layers, nc_hidden * 4


def light_downsampling(in_nc, nc_hidden, **rand_args):
    downsampling_layers = [RandConv2d(in_nc, nc_hidden, 3, 1, **rand_args)]
    return downsampling_layers, nc_hidden


class BayesianClassifier(nn.Module):
    def __init__(
        self,
        n_scale,
        nclass,
        nc,
        nc_hidden,
        grid_size,
        T,
        downsampling_type="heavy",
        version="v1",
        **rand_args,
    ):
        super().__init__()
        
        if downsampling_type == "light":
            layers, nc_hidden = light_downsampling(nc, nc_hidden, **rand_args)
            self.downsampling_layers = nn.Sequential(*layers)
        elif downsampling_type == "heavy":
            layers, nc_hidden = heavy_downsampling(nc, nc_hidden, **rand_args)
            self.downsampling_layers = nn.Sequential(*layers)
        else:
            raise ValueError("Invalid value of downsampling_type")
        flows = []
        for _ in range(n_scale):
            flow_fn = RandFlowFn(nc_hidden, **rand_args) if version == "v1" else RandFlowFn_v2(nc_hidden, **rand_args)
            flows.append(IntegratedFlow(flow_fn, None, grid_size, T))
        self.multiscale_flows = MultiScaleFlow(flows)
        self.fc_layers = nn.Sequential(
            rand_norm(nc_hidden, **rand_args),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            RandLinear(nc_hidden, nclass, **rand_args),
        )

    def forward(self, x):
        out = self.downsampling_layers(x)
        out = self.multiscale_flows(out)
        out = self.fc_layers(out)
        return out


if __name__ == "__main__":
    rand_args = {
        'sigma_0': 1.0,
        'N': 100,
        'init_s': 1.0,
        'alpha': 0.01,
    }
    classifier = BayesianClassifier(
        n_scale=3, nclass=10, nc=3, nc_hidden=64, grid_size=0.1, T=1.0, **rand_args,
    )
    x = torch.randn(13, 3, 32, 32)
    out = classifier(x)
    print(out.size())

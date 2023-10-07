# Efficient Graphics Representation with Differentiable Indirection
### <i>In SIGGRAPH ASIA '23 Conference Proceedings</i>

[Webpage](https://sayan1an.github.io/din.html)

[Paper + Supplemental](https://arxiv.org/abs/2309.08387)

# Directory structure

```text
├── DifferentiableIndirection
│   └── disneyFit
│   └── textureCompression
├── DifferentiableIndirectionData
│   └── gBuffer
│   └── textureCache
├── DifferentiableIndirectionOutput
```

# Important files

* `DifferentiableIndirection/networksBase.py` -- Defines differentiable arrays `SpatialGrid2D, SpatialGrid3D, and SpatialGrid4D`.
* `DifferentiableIndirection/disneyFit/networks.py` -- Defines <i>Disney BRDF</i> approximation network.
* `DifferentiableIndirection/textureCompression/networks.py` -- Defines texture compression networks with varying `2D, 3D, 4D` cascaded arrays.

# A simple <i>differentiable indirection</i> example

```
import networksBase as nb
import torch

class DifferentiableIndirection(torch.nn.Module):
    def __init__(self, primarySize, cascadedSize, torchDevice):
        super(DifferentiableIndirection, self).__init__()

        # initialize primary - gpu device, array resolutions, channel count, bilinear interpolation,
        # normalize o/p with non-linearity, scale initial content, initialize with uniform ramp - 'U'.        
        self.primary = nb.SpatialGrid2D(torchDevice, uDim=primarySize, vDim=primarySize,
                                      latent=2, bilinear=True, normalize=True, initScale=1, initMode="U")

        # initialize cascaded - gpu device, array resolutions, channel count, bilinear interpolation,
        # no o/p with non-linearity, scale initial content, initialize with constant value. 
        self.cascaded = nb.SpatialGrid2D(torchDevice, uDim=cascadedSize, vDim=cascadedSize,
                                      latent=1, bilinear=True, normalize=False, initScale=0.5, initMode="C")

    # Assumes x \in [0, 1)
    def forward(self, x):
        return self.cascaded(self.primary(x))
```

# Training and inference

Download the training data and place it in the directory structure as outlined above in folder `DifferentiableIndirectionData`. The training and inference output is accumulated in the folder `DifferentiableIndirectionOutput`.

<b>Training <i>Disney BRDF</i> using cascaded-decoders of size 16.</b>
```
cd DifferentiableIndirection/disneyFit
../disneyFit>python .\train.py 16 16 16
```



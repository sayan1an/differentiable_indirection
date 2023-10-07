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

# Training and inference




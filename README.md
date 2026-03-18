<div align="center">
  <a href=https://xenon-w.github.io/FilterGS.github.io/>
    <img src="docs/filtergs_logo.jpg" alt="logo" width="100%" align="center"/>
  </a>
</div>

---

<!-- ![python](https://img.shields.io/github/languages/top/zju3dv/LoG)
![star](https://img.shields.io/github/stars/zju3dv/LoG)
[![license](https://img.shields.io/badge/license-zju3dv-white)](license) -->

**FilterGS** utilizes a single RTX 4090 for training highly realistic urban-scale models and for their real-time rendering. Visit our [**project page**](https://xenon-w.github.io/FilterGS.github.io/) for more demos.

Our code is built upon PyTorch and leverages [gaussian-splatting](https://github.com/graphdeco-inria/diff-gaussian-rasterization) and [LoG](https://github.com/zju3dv/LoG) techniques.

## Quick Start & Dataset Preparation

For a smooth setup, follow the [installation guide](./docs/install.md).

We employ [Colmap](https://colmap.github.io/) to prepare the dataset. Refer to the [preprocessing documentation](./docs/preprocess.md) for detailed instructions.

## Training

Training the model is as simple as one command:

```bash
python3 apps/train.py --cfg config/GauUScene/college/train.yml split train
```

We automatically configure heuristic parameters based on the dataset size.

## Rendering

Before rendering the model, it is necessary to calculate the ancestor_path attribute of the model, which will take about 5 minutes:
```bash
python apps/precompute_ancestor.py --ckpt /PATH/TO/YOUR/MODEL.pth --out /OUTPUT/PATH/model_ancestor.pth
```

Rendering the model is also as simple as one command:

```bash
python3 apps/render.py --cfg config/GauUScene/college/render.yml split train
```
`--skip-save` and `--debug` are optional parameters; `--skip-save` skips saving rendered images, while `--debug` outputs detailed parameters of the rendering process.

## Acknowledgements

We acknowledge the following inspirational prior work:

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Level of Gaussian](https://github.com/zju3dv/LoG)
- [MatrixCity dataset](https://github.com/city-super/MatrixCity)
- [UrbanScene dataset](https://saliteta.github.io/CUHKSZ_SMBU/)
- [GauUScene dataset](https://github.com/RingoWRW/UAVD4L)

More details to be added later.
If you have any questions, please feel free to point them out in the issue section.

Contributions are warmly welcomed! If you've made significant progress on any of these fronts, please consider submitting a pull request.


## Citation

Our paper will be available within a week. We greatly appreciate your early interest in this work!

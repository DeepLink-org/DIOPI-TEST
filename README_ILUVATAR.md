## Run on Iluvatar env.
### Environment Preparation  
1. Make sure Iluvatar release package is installed correctly, and environment variables are exported.
If package installed in default path, then export following:
```
export PATH="/usr/local/corex/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/corex/lib:$LD_LIBRARY_PATH"
```
2. Install torch and cmake from Iluvatar SDK are needed.
Torch: use pip to install torch wheel package, release 2.3 version for example.
```
pip3 install torch-1.10.2+corex.2.3.0-cp38-cp38-linux_x86_64.whl 
```
CMake: Run Iluvatar cmake installation script.
```
bash cmake-3.21.5-corex.2.3.0-linux-x86_64.sh --skip-license --prefix=/usr/local
```
3. `git submodule update --init` to update submodule.
### Build 
We support auto-build script on Iluvatar env. `bash build.sh torch` for torch impl, or `bash build.sh cuda` for cuda impl.
More details on build, please refer to README.md of this repo.
`bash clean.sh` to clean up all built files.
### Test
Iluvatar does not support FP64, so add `--filter_dtype=float64` option for `python/main.py` when do testing.
We add another config files for Resnet50 model testing (`python/conformance/diopi_configs_resnet50v1.py`), and to switch between, comment or uncomment in`python/conformance/__init__.py`:
```
#from .diopi_configs import diopi_configs
from .diopi_configs_resnet50v1 import diopi_configs
```
Test procedure keeps the same.

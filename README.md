# FaceML

## Run with the NVIDIA GPU

From PowerShell in the project folder:

```powershell
.\run-gpu.cmd
```

This launches `main.py` using Ubuntu 24.04 in WSL2 and the project's
`/opt/faceml/.venv` environment. TensorFlow can then use the RTX 5070 Ti Laptop GPU.
The environment lives on Ubuntu's native filesystem for faster installation
and imports; the source files remain in the Windows project folder.

The GPU environment uses the official **prerelease**
`tf-nightly==2.22.0.dev20260920` with CUDA dependencies. This build was chosen
for the recent [Blackwell XLA fix](https://github.com/tensorflow/tensorflow/pull/127013).
The version and direct dependencies are recorded in `requirements-gpu.txt`.
The complete tested package set is saved in `requirements-gpu-lock.txt`.
Do not install the stable `tensorflow` package into this same environment:
both packages provide the `tensorflow` Python module.

To verify GPU calculations, convolution, backpropagation, and XLA compilation:

```powershell
wsl -d Ubuntu-24.04 --cd "$PWD" --exec /opt/faceml/.venv/bin/python verify_gpu.py
```

In an editor connected to WSL, select `/opt/faceml/.venv/bin/python` as the interpreter.
The project lives at `/mnt/c/Users/owner/Documents/GitHub/FaceML` in Ubuntu.

## Run on Windows with the CPU

The Windows environment uses Python 3.12 and stable TensorFlow 2.21.0.

```powershell
.\.venv\Scripts\python.exe main.py
```

For Windows editing, select `.venv\Scripts\python.exe` as the interpreter.
Using this interpreter directly also works when PowerShell blocks activation scripts.

## Recreate the Python environments

Windows, using an installed Python 3.12 interpreter:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ubuntu/WSL, from the project directory:

```bash
python3 -m venv /opt/faceml/.venv
/opt/faceml/.venv/bin/python -m pip install -r requirements-gpu-lock.txt
/opt/faceml/.venv/bin/python verify_gpu.py
```

Ubuntu needs `python3-venv`, `libgl1`, and `libglib2.0-0`; these were installed
during setup. This Ubuntu installation currently uses its default root user;
recreating the environment under `/opt` requires write access there.
The Windows virtual environment is excluded from Git, and the GPU environment
is outside the repository.

The old `tensorflow-gpu` package is retired. Modern TensorFlow uses CUDA
extras inside Linux/WSL2 for NVIDIA GPU support on Windows; the Windows
`.venv` uses the CPU. See the [official installation guide](https://www.tensorflow.org/install/pip).

## Verification completed

The Windows CPU calculation and `main.py` run passed. In WSL, GPU matrix
multiplication, convolution, backpropagation, a parameter update, and XLA
compilation all passed on the RTX 5070 Ti Laptop GPU. The Windows GPU launcher
also ran `main.py` successfully.

This nightly may print a warning about compiling Blackwell kernels from PTX.
The checks above passed with that warning; new GPU workloads can have a slower
first run while kernels are compiled and cached.

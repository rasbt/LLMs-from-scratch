# Native pixi Python and package management

This tutorial is an alternative to the [`./native-uv.md`](native-uv.md) document for those who prefer `pixi`'s native commands over traditional environment and package managers like `conda` and `pip`.

Note that pixi uses `uv add` under the hood, as described in [`./native-uv.md`](native-uv.md).

Pixi and uv are both modern package and environment management tools for Python, but pixi is a polyglot package manager designed for managing not just Python but also other languages (similar to conda), while uv is a Python-specific tool optimized for ultra-fast dependency resolution and package installation.

Someone might choose pixi over uv if they need a polyglot package manager that supports multiple languages (not just Python) or prefer a declarative environment management approach similar to conda. For more information, please visit the official [pixi documentation](https://pixi.sh/latest/).

In this tutorial, I am using a computer running macOS, but this workflow is similar for Linux machines and may work for other operating systems as well.

&nbsp;
## 1. Install pixi

Pixi can be installed as follows, depending on your operating system.

<br>

**macOS and Linux**

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

or

```bash
wget -qO- https://pixi.sh/install.sh | sh
```

<br>

**Windows**

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

> **Note:**
> For more installation options, please refer to the official [pixi documentation](https://pixi.sh/latest/).


&nbsp;
## 1. Install Python

You can install Python using pixi:

```bash
pixi add python=3.10
```

> **Note:**
> I recommend installing a Python version that is at least 2 versions older than the most recent release to ensure PyTorch compatibility. For example, if the most recent version is Python 3.13, I recommend installing version 3.10 or 3.11. You can find out the most recent Python version by visiting [python.org](https://www.python.org).

&nbsp;
## 3. Install Python packages and dependencies

To install all required packages from a `pixi.toml` file (such as the one located at the top level of this GitHub repository), run the following command, assuming the file is in the same directory as your terminal session:

```bash
pixi install
```

> **Note:**
> If you encounter issues with dependencies (for example, if you are using Windows), you can always fall back to pip: `pixi run pip install -U -r requirements.txt`

By default, `pixi install` will create a separate virtual environment specific to the project.

You can install new packages that are not specified in `pixi.toml` via `pixi add`, for example:

```bash
pixi add packaging
```

And you can remove packages via `pixi remove`, for example,

```bash
pixi remove packaging
```

&nbsp;
## 4. Run Python code

Your environment should now be ready to run the code in the repository.

Optionally, you can run an environment check by executing the `python_environment_check.py` script in this repository:

```bash
pixi run python setup/02_installing-python-libraries/python_environment_check.py
```

<br>

**Launching JupyterLab**

You can launch a JupyterLab instance via:

```bash
pixi run jupyter lab
```


---

Any questions? Please feel free to reach out in the [Discussion Forum](https://github.com/rasbt/LLMs-from-scratch/discussions).

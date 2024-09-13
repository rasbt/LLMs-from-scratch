# Python Setup Tips



There are several different ways you can install Python and set up your computing environment. Here, I am illustrating my personal preference.

(I am using computers running macOS, but this workflow is similar for Linux machines and may work for other operating systems as well.)


<br>
<br>


## 1. Download and install Miniforge

Download miniforge from the GitHub repository [here](https://github.com/conda-forge/miniforge).

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/download.png" alt="download" width="600px">

Depending on your operating system, this should download either an `.sh` (macOS, Linux) or `.exe` file (Windows).

For the `.sh` file, open your command line terminal and execute the following command

```bash
sh ~/Desktop/Miniforge3-MacOSX-arm64.sh
```

where `Desktop/` is the folder where the Miniforge installer was downloaded to. On your computer, you may have to replace it with `Downloads/`.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/miniforge-install.png" alt="miniforge-install" width="600px">

Next, step through the download instructions, confirming with "Enter".


<br>
<br>


## 2. Create a new virtual environment

After the installation was successfully completed, I recommend creating a new virtual environment called `LLMs`, which you can do by executing

```bash
conda create -n LLMs python=3.10
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/new-env.png" alt="new-env" width="600px">

> Many scientific computing libraries do not immediately support the newest version of Python. Therefore, when installing PyTorch, it's advisable to use a version of Python that is one or two releases older. For instance, if the latest version of Python is 3.13, using Python 3.10 or 3.11 is recommended.

Next, activate your new virtual environment (you have to do it every time you open a new terminal window or tab):

```bash
conda activate LLMs
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/activate-env.png" alt="activate-env" width="600px">

<br>
<br>

## Optional: styling your terminal

If you want to style your terminal similar to mine so that you can see which virtual environment is active,  check out the [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh) project.

<br>
<br>

## 3. Install new Python libraries



To install new Python libraries, you can now use the `conda` package installer. For example, you can install [JupyterLab](https://jupyter.org/install) and [watermark](https://github.com/rasbt/watermark) as follows:

```bash
conda install jupyterlab watermark
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/conda-install.png" alt="conda-install" width="600px">



You can also still use `pip` to install libraries. By default, `pip` should be linked to your new `LLms` conda environment:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/check-pip.png" alt="check-pip" width="600px">

<br>
<br>

## 4. Install PyTorch

PyTorch can be installed just like any other Python library or package using pip. For example:

```bash
pip install torch
```

However, since PyTorch is a comprehensive library featuring CPU- and GPU-compatible codes, the installation may require additional settings and explanation (see the *A.1.3 Installing PyTorch in the book for more information*).

It's also highly recommended to consult the installation guide menu on the official PyTorch website at [https://pytorch.org](https://pytorch.org).

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/pytorch-installer.jpg" width="600px">


## 5. Installing Python packages and libraries used in this book

Please refer to the [Installing Python packages and libraries used in this book](../02_installing-python-libraries/README.md) document for instructions on how to install the required libraries.

<br>

---




Any questions? Please feel free to reach out in the [Discussion Forum](https://github.com/rasbt/LLMs-from-scratch/discussions).

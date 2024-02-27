# Docker Environment Setup Guide

The notebooks can be run and developed in a docker container without the need to install any software packages on your local machine. This guide will walk you through that process.


## Download and install Docker

The easiest way to get started with docker is by installing [Docker Desktop](https://docs.docker.com/desktop/) for your relevant platform.

Linux (Ubuntu) users may prefer to install the [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) instead and follow the [post installation](https://docs.docker.com/engine/install/linux-postinstall/) steps.


## Install VSCode

Whilst devcontainers work with other IDEs as well, I prefer to use Visual Studio Code. [Install](https://code.visualstudio.com/download) it if you don't have it and want to use it. The instructions below will be VSCode specific but a similar process should apply to PyCharm as well.

1. Clone and `cd` into the project root directory.
2. Type `code .` in the terminal to open the project in VSCode. Alternatively, you can launch VSCode and select the project to open from the UI.
3. Install the **Remote Development** extension from the Extensions tab.
4. Since the `.devcontainer` folder is present, VSCode should automatically detect it and ask whether you would like to open the project in a devcontainer. If it doesn't, simply press `Ctrl + Shift + P` to open the command palette and start typing `dev containers` to see a list of all Dev Container specific options.
5. Select **Reopen in Container**.
6. Once the image has been pulled and built, you should have your project mounted inside the container with all the packages installed, ready for development. 
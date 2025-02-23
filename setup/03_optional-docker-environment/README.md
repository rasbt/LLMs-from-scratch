# Docker Environment Setup Guide

If you prefer a development setup that isolates a project's dependencies and configurations, using Docker is a highly effective solution. This approach eliminates the need to manually install software packages and libraries and ensures a consistent development environment.

This guide will walk you through the process for setting up an optional docker environment for this book if you prefer it over using the conda approach explained in [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) and [../02_installing-python-libraries](../02_installing-python-libraries).

<br>

## Downloading and installing Docker

The easiest way to get started with Docker is by installing [Docker Desktop](https://docs.docker.com/desktop/) for your relevant platform.

Linux (Ubuntu) users may prefer to install the [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) instead and follow the [post-installation](https://docs.docker.com/engine/install/linux-postinstall/) steps.

<br>

## Using a Docker DevContainer in Visual Studio Code

A Docker DevContainer, or Development Container, is a tool that allows developers to use Docker containers as a fully-fledged development environment. This approach ensures that users can quickly get up and running with a consistent development environment, regardless of their local machine setup.

While DevContainers also work with other IDEs, a commonly used IDE/editor for working with DevContainers is Visual Studio Code (VS Code). The guide below explains how to use the DevContainer for this book within a VS Code context, but a similar process should also apply to PyCharm. [Install](https://code.visualstudio.com/download) it if you don't have it and want to use it.

1. Clone this GitHub repository and `cd` into the project root directory.

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
```

2. Move the `.devcontainer` folder from `setup/03_optional-docker-environment/` to the current directory (project root).

```bash
mv setup/03_optional-docker-environment/.devcontainer ./
```

3. In Docker Desktop, make sure that **_desktop-linux_ builder** is running and will be used to build the Docker container (see _Docker Desktop_ -> _Change settings_ -> _Builders_ -> _desktop-linux_ -> _..._ -> _Use_)

4. If you have a [CUDA-supported GPU](https://developer.nvidia.com/cuda-gpus), you can speed up the training and inference:

    4.1 Install **NVIDIA Container Toolkit** as described [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt). NVIDIA Container Toolkit is supported as written [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2).

    4.2 Add _nvidia_ as runtime in Docker Engine daemon config (see _Docker Desktop_ -> _Change settings_ -> _Docker Engine_). Add these lines to your config:

    ```json
    "runtimes": {
        "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
    ```

    For example, the full Docker Engine daemon config json code should look like that:

    ```json
    {
      "builder": {
        "gc": {
          "defaultKeepStorage": "20GB",
          "enabled": true
        }
      },
      "experimental": false,
      "runtimes": {
        "nvidia": {
          "path": "nvidia-container-runtime",
          "runtimeArgs": []
        }
      }
    }
    ```

    and restart Docker Desktop.

5. Type `code .` in the terminal to open the project in VS Code. Alternatively, you can launch VS Code and select the project to open from the UI.

6. Install the **Remote Development** extension from the VS Code _Extensions_ menu on the left-hand side.

7. Open the DevContainer.

Since the `.devcontainer` folder is present in the main `LLMs-from-scratch` directory (folders starting with `.` may be invisible in your OS depending on your settings), VS Code should automatically detect it and ask whether you would like to open the project in a devcontainer. If it doesn't, simply press `Ctrl + Shift + P` to open the command palette and start typing `dev containers` to see a list of all DevContainer-specific options.

8. Select **Reopen in Container**.

Docker will now begin the process of building the Docker image specified in the `.devcontainer` configuration if it hasn't been built before, or pull the image if it's available from a registry.

The entire process is automated and might take a few minutes, depending on your system and internet speed. Optionally click on "Starting Dev Container (show log)" in the lower right corner of VS Code to see the current built progress.

Once completed, VS Code will automatically connect to the container and reopen the project within the newly created Docker development environment. You will be able to write, execute, and debug code as if it were running on your local machine, but with the added benefits of Docker's isolation and consistency.

> **Warning:**
> If you are encountering an error during the build process, this is likely because your machine does not support NVIDIA container toolkit because your machine doesn't have a compatible GPU. In this case, edit the `devcontainer.json` file to remove the `"runArgs": ["--runtime=nvidia", "--gpus=all"],` line and run the "Reopen Dev Container" procedure again.

9. Finished.

Once the image has been pulled and built, you should have your project mounted inside the container with all the packages installed, ready for development.

<br>

## Uninstalling the Docker Image

Below are instructions for uninstalling or removing a Docker container and image if you no longer plan to use it. This process does not remove Docker itself from your system but rather cleans up the project-specific Docker artifacts.

1. List all Docker images to find the one associated with your DevContainer:

```bash
docker image ls
```

2. Remove the Docker image using its image ID or name:

```bash
docker image rm [IMAGE_ID_OR_NAME]
```

<br>

## Uninstalling Docker

If you decide that Docker is not for you and wish to uninstall it, see the official documentation [here](https://docs.docker.com/desktop/uninstall/) that outlines the steps for your specific operating system.

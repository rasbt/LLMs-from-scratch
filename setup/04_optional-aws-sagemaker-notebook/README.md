# AWS CloudFormation Template: Jupyter Notebook with LLMs-from-scratch Repo

This CloudFormation template creates a GPU-enabled Jupyter notebook in Amazon SageMaker with an execution role and the LLMs-from-scratch GitHub repository.

## What it does:

1. Creates an IAM role with the necessary permissions for the SageMaker notebook instance.
2. Creates a KMS key and an alias for encrypting the notebook instance.
3. Configures a notebook instance lifecycle configuration script that:
   - Installs a separate Miniconda installation in the user's home directory.
   - Creates a custom Python environment with TensorFlow 2.15.0 and PyTorch 2.1.0, both with CUDA support.
   - Installs additional packages like Jupyter Lab, Matplotlib, and other useful libraries.
   - Registers the custom environment as a Jupyter kernel.
4. Creates the SageMaker notebook instance with the specified configuration, including the GPU-enabled instance type, the execution role, and the default code repository.

## How to use:

1. Download the CloudFormation template file (`cloudformation-template.yml`).
2. In the AWS Management Console, navigate to the CloudFormation service.
3. Create a new stack and upload the template file.
4. Provide a name for the notebook instance (e.g., "LLMsFromScratchNotebook") (defaults to the LLMs-from-scratch GitHub repo).
5. Review and accept the template's parameters, then create the stack.
6. Once the stack creation is complete, the SageMaker notebook instance will be available in the SageMaker console.
7. Open the notebook instance and start using the pre-configured environment to work on your LLMs-from-scratch projects.

## Key Points:

- The template creates a GPU-enabled (`ml.g4dn.xlarge`) notebook instance with 50GB of storage.
- It sets up a custom Miniconda environment with TensorFlow 2.15.0 and PyTorch 2.1.0, both with CUDA support.
- The custom environment is registered as a Jupyter kernel, making it available for use in the notebook.
- The template also creates a KMS key for encrypting the notebook instance and an IAM role with the necessary permissions.

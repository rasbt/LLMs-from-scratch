name: Bug report
about: Report errors related to the book content or code
labels: bug
assignees: rasbt
body:
  - type: markdown


  - type: textarea
    attributes:
      label: Bug description
      description: A description of the issue.
      placeholder: |
        Please provide description of what the bug or issue is.
    validations:
      required: true

  - type: dropdown
    id: versions
    attributes:
      label: What operating system are you using?
      description: If applicable, please select the operating system where you experienced this issue
      multiple: true
      options:
        - "macOS"
        - "Linux"
        - "Windows"
    validations:
      required: false


  - type: textarea
    attributes:
      label: Environment
      description: |
        Please provide details about your Python environment via the environment collection script or notebook located at
        https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/02_installing-python-libraries
        You can run the script as follows:
        ```console
        wget https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/setup/02_installing-python-libraries/python_environment_check.py
        python python_environment_check.py
        ```

        You can simply copy and paste the outputs of this script below.
      value: |
        <details>
          <summary>Python environment</summary>

        ```
        # [OK] Your Python version is 3.11.4
        # [OK] torch 2.3.1
        # [OK] jupyterlab 4.2.2
        # [OK] tiktoken 0.7.0
        # [OK] matplotlib 3.9.0
        # [OK] numpy 1.26.4
        # [OK] tensorflow 2.16.1
        # [OK] tqdm 4.66.4
        # [OK] pandas 2.2.2
        # [OK] psutil 5.9.8
        ```

        </details>
    validations:
      required: false

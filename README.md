# Setup Instructions

Follow these steps to clone the repository and set up the required environment.

## 1. Clone the Repository

First, clone the repository:

```bash
git clone --branch gpu https://github.com/minoluke/aso_sys
```

## 2. Create the Conda Environment

Navigate to the cloned directory and create the Conda environment using the environment.yml file:

```bash
cd aso_sys
conda env create -f environment.yml
```

## 3. Activate the Conda Environment

Activate the newly created Conda environment:

```bash
conda activate aso_env
```

## 4. Run the Project

Once the environment is set up, run the main program using:

```bash
python main.py
```

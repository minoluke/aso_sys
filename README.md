# Setup Instructions

Follow these steps to build and run the Docker container for the project.

## 1. Clone the Repository

First, clone the repository:

```bash
git clone https://github.com/minoluke/aso_sys
```

## 2. Navigate to the Project Directory

Change to the project directory:

```bash
cd aso_sys
```

## 3. Build the Docker Image

Build the Docker image using the provided `Dockerfile`:

```bash
docker build -t aso_sys_image .
```

## 4. Run the Docker Container

Run the container with the built image:

```bash
docker run -it --rm -v $(pwd)/data:/workspace/data aso_sys_image
```

This command mounts the local `data` directory to the container's `/workspace/data` directory.

## 5. Execute the Main Script

Inside the container, execute the main program:

```bash
python main.py
```

## Notes

- The `Dockerfile` installs all necessary dependencies, including:
  - Python 3.12
  - Libraries like tsfresh, scikit-learn, and lightgbm
- Ensure Docker is installed and running on your system before executing these steps.
- For GPU support, modify the `Dockerfile` and ensure your environment has the necessary drivers and CUDA toolkit installed.
- Now, only dummy data is provided in this repository due to the restriction on data redistribution. When running, please replace data/dummy_observation_data.dat with data/observation_data.dat.
  - We cannot make all data public, but We are currently planning to add code to install actual Hi-net data.
 
## License and Attribution
This project builds upon the work from Dempsey et al. (2020), whose original code was released under the MIT License.

Original Repository: https://github.com/ddempsey/whakaari
Modifications & Extensions: We have made adaptations and improvements tailored for our specific use case.
We acknowledge and appreciate the original work, and all modifications follow the terms of the MIT License.

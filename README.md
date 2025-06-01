# Setup Instructions

Follow the steps below to build and run the Docker container for this project.

## 1. Clone the Repository

```bash
git clone https://github.com/minoluke/aso_sys
cd aso_sys
```

## 2. Build the Docker Image

```bash
docker build -t aso_sys_image .
```

## 3. Run the Docker Container

```bash
docker run -it --rm -v $(pwd)/data:/workspace/data aso_sys_image
```

This command mounts your local `data/` directory to `/workspace/data` inside the container.

## 4. Run the Main Script

Once inside the container:

```bash
python main.py
```

---

# Quick Start

To quickly test the system with minimal setup, run:

```bash
python test_run.py
```

This script will:

- Download 10 minutes of Hi-net waveform data (for testing purposes)
- Train and test a model using fixed parameters

## Parameters

- observation_data_key: Type of data stream to use (e.g., "tremor")

- look_backward: Number of days to look backward from the reference time (e.g., 60.0)

- look_forward: Number of days to look forward to assign eruption labels (e.g., 30.0)

- cv_index: Index used for cross-validation selection (e.g., 0)

- start_period, end_period: Training and testing time window (e.g., "2015-10-09" to "2016-10-09")

## Notes

- In `test_run.py`, replace `'your_username'` and `'your_password'` with valid Hi-net credentials.
- Hi-net Win32tool setup: [https://hinetwww11.bosai.go.jp/auth/manual/?LANG=en](https://hinetwww11.bosai.go.jp/auth/manual/?LANG=en)

## License and Attribution

This project builds upon the work from Dempsey et al. (2020), whose original code was released under the MIT License.

Original Repository: https://github.com/ddempsey/whakaari
Modifications & Extensions: We have made adaptations and improvements tailored for our specific use case.
We acknowledge and appreciate the original work, and all modifications follow the terms of the MIT License.

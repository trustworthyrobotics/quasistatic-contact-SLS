# Quasistatic Contact SLS

This repository contains the code accompanying the paper [*Certified Gradient-Based Contact-Rich Manipulation via Smoothing-Error Reachable Tubes*](https://arxiv.org/abs/2602.09368),
building on top of the [drake](https://drake.mit.edu/) toolbox.

### Install
```sh
# Clone this repository.
git clone https://github.com/trustworthyrobotics/quasistatic-contact-SLS.git

# Change into the project root directory.
cd quasistatic-contact-SLS

# Install the build dependencies.
setup/install_prereqs

# Build and install using standard CMake commands.
mkdir build
cd build
cmake .. -DWITH_USER_FMT=OFF -DWITH_USER_SPDLOG=OFF
make install

# Add packages to python path
export PYTHONPATH=${PWD}/install/lib/python3.12/site-packages:${PYTHONPATH}
```

### Example
The examples are under `examples/planning_through_contact`.

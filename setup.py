from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if (('.png' not in x) and ('.gif' not in x))]
long_description = ''.join(lines)

setup(
    name="robomimic",
    packages=[
        package for package in find_packages() if package.startswith("robomimic")
    ],
    install_requires=[
        "numpy>=1.13.3",
        "h5py",
        "psutil",
        "tqdm",
        "termcolor",
        "tensorboard",
        "tensorboardX",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "egl_probe>=1.0.1",
        "torch==2.0.1",
        "torchvision",
        "diffusers==0.11.1",
        "opencv-python",
        "transformers==4.34.0"
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    author="Suraj Nair, Ashwin Balakrishna, Soroush Nasiriany",
    author_email="ashwin.balakrishna@tri.global",
    version="0.3.0",
    long_description=long_description,
    long_description_content_type='text/markdown'
)

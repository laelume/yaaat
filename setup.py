from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="yaaat",
    version="0.1.0",
    author="laelume",
    description="Yet Another Audio Annotation Tool - Interactive bioacoustics multitool for annotating vocalizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/laelume/yaaat",  
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "librosa",
        "natsort",
        "sounddevice",
    ],
    entry_points={
        'console_scripts': [
            'yaaat=yaaat.changepoint_annotator:main',
        ],
    },
)

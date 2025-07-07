from setuptools import setup, find_packages

setup(
    name="springer_segmentation",
    version="0.1.0",
    description="Segmentation of heart sounds using Springer algorithm",
    author="Your Name",
    packages=find_packages(),  # Sucht automatisch alle Python-Module
    install_requires=[
        "numpy",
        "matplotlib",
        "librosa",
        "scipy",
        "scikit-learn",
        "PyWavelets",
        "soundfile",
        "tqdm",
        "numba",
        "soxr"
    ],
    include_package_data=True,
    package_data={
        "springer_segmentation": ["saved_model.pkl"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
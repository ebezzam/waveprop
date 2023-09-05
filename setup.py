import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="waveprop",
    version="0.0.7",
    author="Eric Bezzam",
    author_email="ebezzam@gmail.com",
    description="Functions and scripts to simulate free-space optical wave propagation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ebezzam/waveprop",
    # download_url = "https://github.com/ebezzam/waveprop/archive/refs/tags/v0.0.1.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.12.1",
        "torchvision>=0.13.1",
        "opencv-python",
        "numpy",
        "scipy",
        "matplotlib",
        "pyffs",
        "progressbar",
    ],
    include_package_data=True,
)

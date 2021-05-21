import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="waveprop",
    version="0.0.1",
    author="Eric Bezzam",
    author_email="ebezzam@gmail.com",
    description="Functions and scripts to simulate free-space optical wave propagation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ebezzam/waveprop",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy==1.19.5", "matplotlib==3.3.4", "scipy==1.5.4"],
)

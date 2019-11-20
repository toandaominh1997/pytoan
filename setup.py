import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytoan",
    version="0.5.5",
    author="toandaominh1997",
    author_email="toandaominh1997@gmail.com",
    description="A library of toandaominh1997",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/toandaominh1997/pytoan",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name = "neural_net_challenge",
        version = "0.0.5",
        author = "Julia Gonik",
        author_email = "jgonik@mit.edu",
        description = "Building a small neural net from scratch",
        long_description = long_description,
        long_description_content_type = "text/markdown",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
)

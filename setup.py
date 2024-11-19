# setup.py
from setuptools import setup, find_packages

setup(
    name="ireason",
    version="0.1.0",
    packages=find_packages("src"),  # Look for packages in 'src'
    package_dir={"": "src"},  # Tell setuptools that packages are under 'src'
    install_requires=[
        "datasets==3.1.0",
        "networkx==3.4.2",
        "pandas==2.2.3",
    ],
    description="A description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jacksonkunde/i-reason",
    author="Jackson Kunde",
    author_email="jacksonkunde@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

from setuptools import find_packages, setup
from typing import List

def get_requirements(filepath: str) -> List[str]:
    """
    this function will return list of requirements from file path
    """
    requirements = []
    with open(filepath) as f:
        requirements= f.readlines()
        requirements = [req.replace("/n","") for req in requirements]
    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements



setup(
    name="mllearn",
    version="0.0.1",
    author="malik",
    author_email="malikidoko@gmail.com",
    packages=find_packages(),
    install_reuires=get_requirements('requirements.txt')
)
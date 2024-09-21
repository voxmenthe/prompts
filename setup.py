from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='prompts',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='Jeff Coggshall',
    author_email='thenextlocalminima@gmail.com',
    description='A brief description of your project',
    url='https://github.com/voxmenthe/prompts',
)
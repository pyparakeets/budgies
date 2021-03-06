import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="threshold_optimizer",
    version="0.0.1a1",
    description="Optimize decision boundary/threshold for predicted probabilities from binary classification",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/pyparakeets/budgies",
    author="Mawuli Adjei, Adu Boahene, Tobelum Eze Okoli, Wayne Yu",
    author_email="pyparakeets@gmail.com",
    license="MIT",
    keywords='Optimize Decision Boundary Threshold Binary Classification Probabilities',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    project_urls={
        # 'Documentation': 'https://benfords-law.readthedocs.io/',
        # 'Funding': 'https://donate.pypi.org',
        # 'Say Thanks!': 'http://saythanks.io/to/example',
        'Source': 'https://github.com/pyparakeets/budgies',
        'Tracker': 'https://github.com/pyparakeets/budgies/issues',
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'scikit-learn==0.24.0',
        'pandas==0.25.1',
        'numpy==1.17.1',
    ],

    python_requires='>=3.6',
)

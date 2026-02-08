from setuptools import setup, find_packages

setup(
    name="composite-machine",
    version="0.1.0a1",
    author="Toni Milovan",
    author_email="tmilovan@fwd.hr",
    description="Automatic calculus via dimensional arithmetic — provenance-preserving composite numbers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tmilovan/composite-machine",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="automatic-differentiation, calculus, Laurent-polynomials, infinitesimals, division-by-zero",
    license="AGPL-3.0",
)

import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="marine",
    version="0.0.1",
    author="Byeongseon Park",
    author_email="6gsn.park@gmail.com",
    description="A unified accent estimation method based on multi-task learning for Japanese text-to-speech",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/6gsn/marine",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
    ],
)
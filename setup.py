from setuptools import setup

setup(
  name='bullinger',
  description='Analyzes video annotations',
  license='Apache',
  packages=['bullinger'],
  zip_safe=False,
  python_requires=">=3.6",
  package_data={
    # If any package contains *.txt or *.rst files, include them:
    "": []
  }
)
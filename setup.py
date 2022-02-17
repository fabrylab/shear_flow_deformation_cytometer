from setuptools import setup


setup(name='shear_flow_deformation_cytometer',
      version="0.1",
      packages=['shear_flow_deformation_cytometer'],
      description='Cell deformation under shear flow analysis package in python.',
      author='Richard Gerum',
      author_email='richard.gerum@fau.de',
      license='GPLv3',
      install_requires=[
            'numpy',
            'matplotlib',
            'scipy',
            'tqdm',
            'tensorflow >= 2.5.3',
            'tensorflow-addons == 0.12.1',
            'scikit-image>=0.17.2',
            'imageio',
            'tifffile',
            "opencv-python",
            'qtawesome>=1.0.0',
            'qimage2ndarray',
            "pandas",
            "h5py",
            "pint",
            "pint_pandas",
            "pyQt5"
      ],
)

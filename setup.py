from setuptools import setup, find_packages
import sys, os.path

setup(name='muscle_gym',
      version='0.0.1',
      description='Muscle Models for Mujoco Agents in Gym',
      url='',
      author='Seungmoon, Rawal',
      author_email='',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('muscle_gym')],
      zip_safe=False,
      install_requires=[
          'numpy>=1.10.4', 'requests>=2.0', 'six', 'pyglet>=1.2.0', 'scipy', 'matplotlib'
      ],
      package_data={'muscle_gym': ['envs/mujoco/assets/*.xml',
                                  'envs/mujoco/muscle_assets/*.yml',
                                  'envs/mujoco_profile/assets/*.xml']},
      tests_require=['pytest', 'mock'],
)
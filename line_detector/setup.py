import os

from setuptools import setup
from glob import glob


package_name = 'line_detector'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, f'{package_name}.include'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nicholas-gs',
    maintainer_email='nicholasganshyan@gmail.com',
    description='Line Detector Node',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_detector_node = line_detector.line_detector_node:main'
        ],
    },
)

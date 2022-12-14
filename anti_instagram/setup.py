import os

from setuptools import setup
from glob import glob

package_name = 'anti_instagram'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nicholas-gs',
    maintainer_email='nicholasganshyan@gmail.com',
    description='Make lane detection robust to illumination variation',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'anti_instagram_node=anti_instagram.anti_instagram_node:main'
        ],
    },
)

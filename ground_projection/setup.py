import os

from setuptools import setup
from glob import glob

package_name = 'ground_projection'

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
    author="Changhyun Choi",
    author_email="cchoi@csail.mit.edu",
    maintainer='nicholas-gs',
    maintainer_email='nicholasganshyan@gmail.com',
    description="""Project the line segments detected in the image to the ground
    plane and in the robot's reference frame
    """,
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ground_projection_node = ground_projection.ground_projection_node:main'
        ],
    },
)

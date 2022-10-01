import os

from setuptools import setup
from glob import glob

package_name = 'fsm'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author="Michael Misha Novitzky",
    author_email="novitzky@mit.edu",
    maintainer='nicholas-gs',
    maintainer_email='nicholasganshyan@gmail.com',
    description='The finite state machine coordinates the modes of the car.'\
        ' The fsm package consists of two nodes, namely `fsm_node` and `logic_gate_node`.',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fsm_node = fsm.fsm_node:main',
            'logic_gate_node = fsm.logic_gate_node:main'
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'geometry_vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@ros2.com',
    description='Distributed Multi-View Geometry Reasoning System',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node     = geometry_vision.camera_node:main',
            'keypoint_node   = geometry_vision.keypoint_node:main',
            'descriptor_node = geometry_vision.descriptor_node:main',
            'matching_node   = geometry_vision.matching_node:main',
            'filtering_node  = geometry_vision.filtering_node:main',
            'geometry_node   = geometry_vision.geometry_node:main',
            'motion_node     = geometry_vision.motion_node:main',
            'decision_node   = geometry_vision.decision_node:main',
        ],
    },
)

from setuptools import setup

package_name = 'vla'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, 'vla_src'],
    package_dir={
        'vla': 'vla',
        'vla_src': 'src'
    },
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Staff Robotics Engineer',
    maintainer_email='staff@autodrive.team',
    description='VLA controller with Apple MLX and safety constraints',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_node = vla.vla_node:main',
        ],
    },
)

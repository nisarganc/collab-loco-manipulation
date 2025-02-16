from setuptools import setup

package_name = "vlm_model"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),        
        ("share/" + package_name, ['package.xml']),
    ],
    install_requires=["setuptools", "msgs_interfaces", "SAM-2"],
    zip_safe=True,
    maintainer="Nisarga Nilavadi",
    maintainer_email="nisarga.nilavadi@utn.de",
    description="ToDo",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "scene_perception = vlm_model.scene_perception:main",
            "GPT_node = vlm_model.GPT_node:main",
        ],
    },
)

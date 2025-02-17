# Scene Aware Zero-Shot Collaborative Loco-Manipulation using Vision-Language Models

## Project dependencies

- **ROS2 Humble**
- **Python and C++ requirements** 

- **GTSAM Repository**
```bash
git clone https://github.com/borglab/gtsam.git
cd gtsam
git checkout 618ac28f2cc407e27e9eaf4e36ece64bc236b8e7
mkdir build && cd build
cmake -DGTSAM_ALLOW_DEPRECATED_SINCE_V42=OFF .. # Disable deprecated functionality for compatibility
sudo make install
echo 'export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/share:${LD_LIBRARY_PATH}' >> ~/.bashrc
source ~/.bashrc
cd ..
cd ..
```

## Project Set-up
```bash
  mkdir coloco-manipulation
  cd coloco-manipulation
  python -m venv objectpushing
  source ~/coloco-manipulation/objectpushing/bin/activate
  git clone https://github.com/nisarganc/coloco-manipulation.git src
  rosdep install --from-paths src --ignore-src -r -y
  colcon build --symlink-install
```

## Run Project
```bash
ros2 launch turtles_bringup coloco_rosnodes.launch.py
ros2 launch turtles_bringup coloco_manipulation.launch.py
```

## Results
![Scene Perception](detection_results.jpg)

## Acknowledgements
- [SAM-2](https://github.com/facebookresearch/sam2)
- [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)
- [ROS-LLM](https://github.com/Auromix/ROS-LLM)
- [MR.CAP](https://github.com/h2jaafar/mr.cap)
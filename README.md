# Among-Us-Detection
 NTU EEE DIP Apr2021
 
 To run Object Detection,
1. clone the repo 
```bash
git clone --recursive https://github.com/andrewivan123/Among-Us-Detection
```
2. Install Tensorflow Object Detection API
```bash 
cd models/research/
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```
3. Install other requirements
```bash
pip install -r requirements.txt
```
4. Run the program
```bash
python object_detection_drone.py
```

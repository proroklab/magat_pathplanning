
# MAgent Guide
1.First clone and compile the MAgent files in another folder

2.Copy folder "build" to utils/magent

3.Run cmd
```
cd project_folder_root
export PYTHONPATH=$(pwd)/utils::$PYTHONPATH
```
4.Run demo
```
python utils/magent_demo.py
```
5.watch visualizer
```
cd utils/magent/build/render
./render
```
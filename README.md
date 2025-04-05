This code relies on python3.6. All packages need to be installed and compatible with 3.6. protobuf must be version 3.19.6.

To run camera inference on the nano, you must use a raspberry pi camera v2.1 for the native IMX219 support (the 2.1 is an imx219 camera)

Running the code is as follows:

python3.6 [ code file name ].py

downloading onnxruntime comes from the jetson zoo. you must then download the wheel with:

python3.6 -m pip install [wheel]

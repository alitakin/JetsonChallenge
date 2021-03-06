# TUT Embedded Smile Detector

The idea is to create and deploy an embedded facial expressions detector application on the Nvidia Jetson TX2. We have developed an application capable of detecting faces and computing probability of smiles on them.
link of challange: [NVIDIA Jetson Challenge](https://developer.nvidia.com/embedded/community/jetson-challenge)
read more about the device: [jetson-tx2](https://developer.nvidia.com/embedded/buy/jetson-tx2)

## Demo 
Link to a youtube video [Short Demo](https://www.youtube.com/watch?v=4JGatQOchFo&feature=youtu.be)

## The workflow:

* Capture frames in the main thread and make them accessible for the other threads.
* Detect multiple faces and assign face coordinates.
* Estimate probability of smiles in face rectangles.
* Sketch the frame with prediction results. 

The sequence diagram looks like this:

![sequence diagram]( https://github.com/alitakin/JetsonChallenge/blob/master/Nvidia_JetSon.PNG)

## Authors
* **Pedram Ghazi**  - pedram.ghazi@tut.fi
* **Saboktakin Hayati**  - saboktakin.hayati@tut.fi
* **Heikki Huttunen**  - huttunen.heikki@tut.fi

## Built With
* [Python 2.7.12](https://www.python.org/download/releases/2.7/)
* [OPEN CV 2.4.13.1](https://opencv.org/)
* [tensorflow 1.3.0](https://github.com/tensorflow/)  
* [keras 2.1.1](https://keras.io/) 
* [numpy 1.11.0](http://www.numpy.org/)   



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


## To Do 
* Apply multithreading for other components and modules.
* Pretrain with other data sets.
* Further development for recognizing other facial expressions.

## Acknowledgments
* [Tampere University of Technology](http://www.tut.fi/en/home).
* [Signal Processing Lab](http://www.tut.fi/en/about-tut/departments/signal-processing/). 
* Hat tip to anyone who's code was used.
* Inspiration.
* etc

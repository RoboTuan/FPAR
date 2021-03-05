**Group Project for the course "Machine Learning and Deep Learning" at Politecnico di Torino**

**Contributors:**<br/>
Davide Bussone<br/>
Antonio Dimitris Defonte<br/>
Filippo Cortese<br/>


# FPAR
Being able to recognize actions and activities from videos is one of the most challenging yet well reasearched tasks in computer vision and its related fields. Applications range from security, behavioural analysis and survaillance to autonomous driving, human computer interaction, automatic indexing and retrieval of media content. Traditionally, researchers have been more focused on action recognition from third person view, but these techniques are less suitable for first person action recognition. In this last setting, wearable cameras are usually head mounted and, as a first issue, we need to address the *egomotion* that may or may not be representative of the performed action. Other relevant challenges include the lack of information about the pose of the performer, the difficulty to understand the object related to the action for the occlusion of the hand or arm and more in general the scarcity of motion information avaiable about the action itself. This challenges are even more relevant if, instead of classifying a general motion pattern (action recognition) like take, put, eat, open, etc., we are are interested in activity recognition, that is identifying both not only the action but also the object involve in it, such as open chocolate, pour water, put sugare in bread, etc.
We will perform this kind of recognition on the [GTEA61](http://cbs.ic.gatech.edu/fpv/) dataset using the [Ego-RNN](https://github.com/swathikirans/ego-rnn) as a staring point. Subsequently, we will move beyond this two-stream network and try other approaches with a self supervised auxiliary pretext motion segmentation task.


## Ego-RNN
This kind of networks has two different branches. The RGB branch consists in a *resnet34* and a *conv-LSTM*. The motion branch instead has only the *resnet34* and exploits the warp flow. After a separate training, we fined tune both branches with a joint training.
We want the network to discriminate among regions that are relevant to the action, so an attetion mechanism is build upon the class activation maps (*CAMs*) with the weights of the fully connected layer of the *resnet34* and the *conv-LSTM* of the RBG branch with those features.

![twoStream7PourBread](https://user-images.githubusercontent.com/57213004/110153747-ba34c200-7de3-11eb-9ec2-a5597403abf1.gif)
![twoStream7OpenChocolate](https://user-images.githubusercontent.com/57213004/110153770-c3259380-7de3-11eb-8aa5-e882a197d774.gif)


## Self-Supervised Task
Now, our aim is to catch a method able to perform the action recognition on a single stream. This wish is realized thanks to an additional branch, called self-supervised branch, which employes a pretext motion segmentation task to interlace spatial and temporal features. 
In general, self-supervised learning is considered as a subset of unsupervised learning in which the networks are trained with an auxiliary pretext task in which pseudo-labels are automatically generated based on some data attributes. As ground-truth labels we exploited the improved dense trajectories, briefly called IDT. The pretext task addresses firstly a classification problem and finally also a regression one. The difference between the two methods are then underlined and analyzed.

![SelfSupClassOpenChocolate](https://user-images.githubusercontent.com/57213004/110154051-14ce1e00-7de4-11eb-8490-5c19e547a450.gif)
![selfSupClassPourBread](https://user-images.githubusercontent.com/57213004/110154086-1dbeef80-7de4-11eb-893b-88894a9ce340.gif)

![RegSelfSupPourBread](https://user-images.githubusercontent.com/57213004/110153950-fec05d80-7de3-11eb-86ca-0af0043f1348.gif)
![RegSelfSupOpenChocolate](https://user-images.githubusercontent.com/57213004/110154021-0c75e300-7de4-11eb-92a6-079e954bfe7b.gif)


## Conv LSTA
After a detailed analysis through the confusion matrices and the visualization of the *CAMs*, we noticed that the models had some pattern in the wrong predictions. The architecture of the *LSTA* can improve the recognition of activities that regard include multiple objects. The *conv-LSTM* is extended with a recurrent attention and with an output pooling which has a high capacity output gate. In this way the attention mechanism is improved so that it can track previously activated regions.

![Lsta7FramesPourBreadChocolate](https://user-images.githubusercontent.com/57213004/110154929-22d06e80-7de5-11eb-9249-b93b76bad01b.gif)
![Lsta7FramesOpenChocolate](https://user-images.githubusercontent.com/57213004/110154966-2bc14000-7de5-11eb-8878-2c7480260403.gif)


## Cross-Attention Modality
Until now spatial and temporal information are fused till the final layer of the network. We tried an approach, whose aim is to let RGB embed information of the flow branch and viceversa, before the generation of spatial attention maps. We selected the 4th layer of the ResNet to perform this. We maintain the two-stream architecture as Ego-RNN section.



## Warp Flow based Self-Supervised Task
At this point we still had some regularities in the errors of the networks, so we decided to try a different motion segmentation task for the model with the self supervised head. We exploited the warp flow and since the respective images are gray-scaled, the regression was the most suitable choice for the network implementation.

![WarpFlowBasedOpenChocolate](https://user-images.githubusercontent.com/57213004/110155323-c02ba280-7de5-11eb-8f04-666e6899c350.gif)
![WarpFlowBasedPourBreadChocolate](https://user-images.githubusercontent.com/57213004/110155365-cc176480-7de5-11eb-8fb7-471558217586.gif)



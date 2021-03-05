**Group Project for the course "Machine Learning and Deep Learning" at Politecnico di Torino**

**Contributors:**<br/>
Davide Bussone<br/>
Antonio Dimitris Defonte<br/>
Filippo Cortese<br/>

# FPAR
Being able to recognize actions and activities from videos is one of the most challenging yet well reasearched tasks in computer vision and its related fields. Applications range from security, behavioural analysis and survaillance to autonomous driving, human computer interaction, automatic indexing and retrieval of media content. Traditionally, researchers have been more focused on action recognition from third person view, but these techniques are less suitable for first person action recognition. In this last setting, wearable cameras are usually head mounted and, as a first issue, we need to address the *egomotion* that may or may not be representative of the performed action. Other relevant challenges include the lack of information about the pose of the performer, the difficulty to understand the object related to the action for the occlusion of the hand or arm and more in general the scarcity of motion information avaiable about the action itself. This challenges are even more relevant if, instead of classifying a general motion pattern (action recognition) like take, put, eat, open, etc., we are are interested in activity recognition, that is identifying both not only the action but also the object involve in it, such as open chocolate, pour water, put sugare in bread, etc.
We will perform this kind of recognition on the [GTEA61](http://cbs.ic.gatech.edu/fpv/) dataset using the [Ego-RNN](https://github.com/swathikirans/ego-rnn) as a staring point. Subsequently, we will move beyond this two-stream network and try other approaches with a self supervised auxiliary pretext motion segmentation task.

## Ego-RNN
This kind of networks has two different branches. The RGB branch consists in a *resnet34* and a *conv-LSTM*. The motion branch instead has only the *resnet34*. After a separate training, we fined tune both branches with a joint training.
We want the network to discriminate among regions that are relevant to the action, so an attetion mechanism is build upon the class activation maps (*CAMs*) with the weights of the fully connected layer of the *resnet34* and the *conv-LSTM* of the RBG branch with those features.

## Self-Supervised Task
Now, our aim is to catch a method able to perform the action recognition on a single stream. This wish is realized thanks to an additional branch, called self-supervised branch, which employes a pretext motion segmentation task to interlace spatial and temporal features. 
In general, self-supervised learning is considered as a subset of unsupervised learning 

## Conv LSTA

## Cross-Attention Modality

## Warp Flow based Self-Supervised Task

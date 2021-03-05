**Group Project for the course "Machine Learning and Deep Learning" at Politecnico di Torino**

**Contributors:**<br/>
Davide Bussone<br/>
Antonio Dimitris Defonte<br/>
Filippo Cortese<br/>

# FPAR
Being able to recognize actions and activities from videos is one of the most challenging yet well reasearched tasks in computer vision and its related fields. Applications range from security, behavioural analysis and survaillance to autonomous driving, human computer interaction, automatic indexing and retrieval of media content. Traditionally, researchers have been more focused on action recognition from third person view, but these techniques are less suitable for first person action recognition. In this last setting, wearable cameras are usually head mounted and, as a first issue, we need to address the *egomotion* that may or may not be representative of the performed action. Other relevant challenges include the lack of information about the pose of the performer, the difficulty to understand the object related to the action for the occlusion of the hand or arm and more in general the scarcity of motion information avaiable about the action itself. This challenges are even more relevant if, instead of classifying a general motion pattern (action recognition) like take, put, eat, open, etc., we are are interested in activity recognition, that is identifying both not only the action but also the object involve in it, such as open chocolate, pour water, put sugare in bread, etc.
We will perform this kind of recognition on the [GTEA61](http://cbs.ic.gatech.edu/fpv/) dataset using the [Ego-RNN](https://github.com/swathikirans/ego-rnn) as a staring point. Subsequently, we will move beyond this two-stream network and try other approaches with a self supervised auxiliary pretext motion segmentation task.

## Ego-RNN

## Self-supervised task

## Conv LSTA

## Cross-attention modality

## Warp flow based self-supervised task

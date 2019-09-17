# YetAnotherMIDIGenerator
end-to-end MIDI Generator using Neural Networks

## Design
I want to build a music generator with a few constraints:
- It has to be able to run on my laptop, not some fancy infrastructure (even for training)
- Works with a small dataset (i.e. my favorite music)
- The tool should be end-to-end (from process training data to output the actual music)
- The process should be divided into different self-contained phases so I can run each phase whenever I feel like it

Those constraints lead to these decisions:
- The MIDI format: music can be (overly simplified) characterized by pitch, duration, and tone color. With MIDI, my model can focus on only pitch and duration if I just focus on 1 channel - the piano channel (i.e. 1 less dimension to care about) => Simpler model
- The model should be as simple as possible
- Each note combination is mapped to a number, the output of the current X note combinations is the next note combination played in the MIDI => the numerical value is just a label, magnitude means nothing => model as a classification problem
- Also, I was not being scientific and wanted to do this purely based on "feeling", hence no val/test set

## Model Summary
For 208242 different note combinations
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 40)]              0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 40, 64)            13327488  
_________________________________________________________________
dense_3 (Dense)              (None, 40, 128)           8320      
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 40, 128)           0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 256)               197376    
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_5 (Dense)              (None, 208242)            53518194  
=================================================================
Total params: 67,117,170
Trainable params: 67,117,170
Non-trainable params: 0
```

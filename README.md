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
For 123645 different note combinations
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 40)]              0
_________________________________________________________________
reshape (Reshape)            (None, 40, 1)             0
_________________________________________________________________
conv1d (Conv1D)              (None, 37, 64)            320
_________________________________________________________________
flatten (Flatten)            (None, 2368)              0
_________________________________________________________________
dense (Dense)                (None, 128)               303232
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 256)               33024
_________________________________________________________________
dropout (Dropout)            (None, 256)               0
_________________________________________________________________
dense_2 (Dense)              (None, 123646)            31777022
=================================================================
Total params: 32,113,598
Trainable params: 32,113,598
Non-trainable params: 0
_________________________________________________________________
```

The use of `pretty_midi` to generate output is heavily influenced by https://gist.github.com/haryoa/a3b969109592fafd370ddf377b4ea0e4#file-gist-py

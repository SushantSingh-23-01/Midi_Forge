# Midi_Forge

## Overview

**Topic**: Midi-Forge is a project that uses a modern Large Language Model (LLM) architecture to generate MIDI files.

 ## Description
 
Midi-Forge aims to leverage the power of advanced LLM architecture to create musical compositions. The model is trained on a dataset consisting of both scraped and manually transcribed MIDI files, ensuring a diverse range of musical structures and styles. 

## Tokenization

The tokenization process in Midi-Forge is designed to capture essential musical elements, including:

1. **Pitch**: The specific note or pitch value.
2. **Relative Position**: The timing within a measure or beat structure.
3. **Duration**: The length or duration of each note.
This approach ensures that the generated MIDI files maintain a coherent and structured musical form.

## Architecture
The architecture of Midi-Forge is based on the LLaMA series, with modifications to enhance its performance. Key features include:
1. **Pre-Layer Norm**: This modification helps in stabilizing training dynamics
2. **Rotary Positional Encoding**: Provides more accurate positional information for long-range dependencies.
3. **Swish GLU Activation Function**: Enhances non-linearity and model expressiveness.
4. **Group Query Attention**: Improves the efficiency and effectiveness of attention mechanisms.

## Decoding Strategies
Midi-Forge employs various decoding strategies to generate MIDI files:
1. **Greedy-Generation**
2. **Top-k Sampling**
3. **Top-p Sampling**

## Examples
### Demo Audio sample

https://github.com/user-attachments/assets/562fbc53-9ea9-4cde-aeb7-2f6684d927e6


# Spiking Neural Network
Experimenting with SNN's to develop novel architectures and learn about brain


## Input
The input is encoded into a readable format that the network understands - A Spike Train 🚂. 
Spike Trains are binary representations over time which are stochastic. The hyperparamater `T` controls the amount of timesteps for the model. However the timesteps are more based on the input. For the MNIST example, pixels are mapped into probabilities which are executed over `T` timesteps. Each of the pixels are fed into the network one timestep at a time. They are binary meaning the signal by itself carries no meaning - rather the time (or number of steps) bewtween the spikes since the model updates based on decaying values.

## Network
The core of a SNN looks similar to traditional ANN's however are fundementally very different. The network is made up of layers which are sparesly connected to neurons in the previous layer. In *normal* ANN's neurons store continous floating point values inside themselves. SNN's however only have discrete binary neurons. Again, we are leveraging the **dimension** of time to express information. Since each neuron is a 1 or a 0, the weights are the driving force.

## Architecture

## Goal 1:
Demonstrate spiking neural networks in traditional settings
- Achieve 95%+ on MNIST
- Replicate already existing models that work


## Goal 2:
Push architecture to its limit and discover bounds
- Tune model to optimal ability
- Provide correct data

## Goal 3:
Expand architecture to multi-modal, online and continous learning
- Develop novel techniques for a general purpose model
- Keep it learning forever, add RL, higher level reward functions


## Referenes/Inspiration/Links
- [Shikhargupta/Spiking-Neural-Network](https://github.com/Shikhargupta/Spiking-Neural-Network)


- PDF: https://binds.cs.umass.edu/pdfs/stdp.pdf


- Paper: [Surrogate Gradient Learning in Spiking Neural Networks](https://arxiv.org/abs/1901.09948)
- PDF: https://arxiv.org/pdf/1901.09948


- Paper: [Self-organization of multi-layer spiking neural networks](https://arxiv.org/abs/2006.06902)
- PDF: https://arxiv.org/pdf/2006.06902


- Paper: [Unsupervised learning of digit recognition using spike-timing-dependent plasticity](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full)
- PDF: https://public-pages-files-2025.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/pdf

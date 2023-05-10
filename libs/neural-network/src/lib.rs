use rand::prelude::*;

pub struct Network{
    layers: Vec<Layer>,
}

struct Layer{
    neurons: Vec<Neuron>,
}

struct Neuron{
    bias: f32,
    weights: Vec<f32>,
}

pub struct LayerTopology{
    pub neurons: usize,
}

impl Network{
    pub fn random(rng: &mut dyn rand::RngCore, layers: &[LayerTopology]) -> Self{
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers| {
                Layer::random(rng, layers[0].neurons, layers[1].neurons)
            })
            .collect();

        Self { layers }
    }

    pub fn propagate(&self, mut inputs: Vec<f32>) -> Vec<f32>{
        for layer in &self.layers{
            inputs = layer.propagate(inputs);
        }

        inputs
    }
}

impl Layer{
    pub fn random(rng: &mut dyn rand::RngCore, input_neurons: usize, output_neurons: usize,) -> Self{
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();

        Self {neurons}
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32>{
        let mut outputs = Vec::with_capacity(self.neurons.len());

        for neuron in &self.neurons{
            let output = neuron.propagate(&inputs);
            outputs.push(output);
        }

        outputs
    }
}

impl Neuron{
    pub fn random(rng: &mut dyn rand::RngCore, output_size: usize) -> Self{
        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..output_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self {bias, weights}
    }

    fn propagate(&self, inputs: &[f32]) -> f32{
        assert_eq!(inputs.len(), self.weights.len());

        let mut output = 0.0;

        for i in 0..inputs.len(){
            output += inputs[i] * self.weights[i];
        }

        output += self.bias;

        (output).max(0.0)
    }
}

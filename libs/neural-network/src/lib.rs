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
    pub fn random(layers: &[LayerTopology]) -> Self{
        assert!(layers.len() > 1);

        let mut built_layers = Vec::new();
    
        for adjacent_layers in layers.windows(2){
            let input_neurons = adjacent_layers[0].neurons;
            let output_neurons = adjacent_layers[1].neurons;
    
            built_layers.push(Layer::random(input_neurons, output_neurons,));
        }

        Self {layers: built_layers}
    }

    pub fn propogate(&self, mut inputs: Vec<f32>) -> Vec<f32>{
        for layer in &self.layers{
            inputs = layer.propogate(inputs);
        }

        inputs
    }
}

impl Layer{
    pub fn random(input_neurons: usize, output_neurons: usize,) -> Self{
        let mut neurons = Vec::new();

        for _ in 0..output_neurons{
            neurons.push(Neuron::random(input_neurons));
        }

        Self {neurons}
    }

    fn propogate(&self, inputs: Vec<f32>) -> Vec<f32>{
        let mut outputs = Vec::with_capacity(self.neurons.len());

        for neuron in &self.neurons{
            let output = neuron.propogate(&inputs);
            outputs.push(output);
        }

        outputs
    }
}

impl Neuron{
    pub fn random(output_size: usize) -> Self{
        let mut rng = rand::thread_rng();

        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..output_size).map(|_| rng.gen_range(-1.0..=1.0)).collect();

        Self {bias, weights}
    }

    fn propogate(&self, inputs: &[f32]) -> f32{
        assert_eq!(inputs.len(), self.weights.len());

        let mut output = 0.0;

        for i in 0..inputs.len(){
            output += inputs[i] * self.weights[i];
        }

        output += self.bias;

        (output).max(0.0)
    }
}

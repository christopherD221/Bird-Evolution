use std::ops::Index;
use rand::RngCore;
use rand::Rng;
use rand::seq::SliceRandom;


//Structures\\
pub struct RouletteWheelSelection;
pub struct UniformCrossover;

pub struct Chromosome{
    genes: Vec<f32>,
}

pub struct GeneticAlgorithm<S>{
    selection_method: S,
    crossover_method: Box<dyn CrossoverMethod>,
    mutation_method: Box<dyn MutationMethod>,
}

pub struct GaussianMutation{
    /// Probability of changing a gene:
    /// - 0.0 = no genes will be touched
    /// - 1.0 = all genes will be touched
    chance: f32,

    /// Magnitude of that change:
    /// - 0.0 = touched genes will not be modified
    /// - 3.0 = touched genes will be += or -= by at most 3.0
    coeff: f32,
}


//Traits\\
pub trait Individual{
    fn chromosome(&self) -> &Chromosome;
    fn fitness(&self) -> f32;
    fn create(chromosome: Chromosome) -> Self;
}

pub trait SelectionMethod{
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I],) -> &'a I
    where
        I: Individual;
}

pub trait CrossoverMethod{
    fn crossover(&self, rng: &mut dyn RngCore, parent_a: &Chromosome, parent_b: &Chromosome) -> Chromosome;
}

pub trait MutationMethod {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome);
}

//Implementation\\
impl GaussianMutation {
    pub fn new(chance: f32, coeff: f32) -> Self {
        assert!(chance >= 0.0 && chance <= 1.0);

        Self { chance, coeff }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };

            if rng.gen_bool(self.chance as _) {
                *gene += sign * self.coeff * rng.gen::<f32>();
            }
        }
    }
}

impl UniformCrossover{
    pub fn new() -> Self{
        Self
    }
}

impl CrossoverMethod for UniformCrossover{
    fn crossover(&self, rng: &mut dyn RngCore, parent_a: &Chromosome, parent_b: &Chromosome) -> Chromosome {
        assert_eq!(parent_a.len(), parent_b.len());
        
        let parent_a = parent_a.iter();
        let parent_b = parent_b.iter();

        parent_a
            .zip(parent_b)
            .map(|(&a, &b)| if rng.gen_bool(0.5) { a } else { b })
            .collect()
        }
}

impl Chromosome{
    pub fn len(&self) -> usize{
        self.genes.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32>{
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32>{
        self.genes.iter_mut()
    }
}

impl Index<usize> for Chromosome {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self {genes: iter.into_iter().collect(),}
    }
}

impl IntoIterator for Chromosome {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

impl RouletteWheelSelection{
    pub fn new() -> Self{
        Self
    }
}

impl SelectionMethod for RouletteWheelSelection{
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I],) -> &'a I
        where
            I: Individual,
    {
            population.choose_weighted(rng, |individual| individual.fitness()).expect("got an empty population")
    }
}

impl<S> GeneticAlgorithm<S> where S: SelectionMethod{
    pub fn new(selection_method: S, crossover_method: impl CrossoverMethod + 'static, mutation_method: impl MutationMethod + 'static) -> Self{
        Self {selection_method, crossover_method: Box::new(crossover_method), mutation_method: Box::new(mutation_method),}
    }

    pub fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I],) -> Vec<I> where I: Individual{
        assert!(!population.is_empty());

        (0..population.len())
        .map(|_| {
            let parent_a = self
                .selection_method
                .select(rng, population)
                .chromosome();

            let parent_b = self
                .selection_method
                .select(rng, population)
                .chromosome();

            let mut child = self
                .crossover_method
                .crossover(rng, parent_a, parent_b);

            self.mutation_method.mutate(rng, &mut child);

            I::create(child)
        })
        .collect()
    }
}

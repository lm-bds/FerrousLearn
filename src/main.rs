/// Trait for that provides description for models and statistical functions
pub trait FerrousLearn {
    // Sigmoid function
    //! Can intake data inside DataShape enum as a vector or vector of vectors
    //
}

#[derive(Debug, Clone)]
enum DataShape {
    Vector(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
}
struct LinearRegression {
    weights: Option<Vec<f64>>,
    learning_rate: f64,
    iterations: usize,
}

impl LinearRegression {
    fn new(learning_rate: f64, iterations: usize) -> LinearRegression {
        LinearRegression {
            weights: None,
            learning_rate,
            iterations,
        }
    }
    fn fit(&mut self, data: &DataShape, target: &Vec<f64>, verbose: bool) {
        let input_size = match &data {
            DataShape::Vector(vector) => 1,
            DataShape::Matrix(matrix) => matrix[0].len(),
        };

        let X: Vec<Vec<f64>> = add_bias(&data);

        self.weights = Some(vec![0.0; input_size + 1]);

        for i in 0..self.iterations {
            let mut gradients = vec![0.0; self.weights.as_ref().unwrap().len()];
            let mut loss = 0.0;
            for (X_row, &target) in X.iter().zip(target.iter()) {
                let predicted: f64 = X_row
                    .iter()
                    .zip(self.weights.as_ref().unwrap())
                    .map(|(x, y)| x * y)
                    .sum();
                let error: f64 = predicted - target;
                loss += error.powi(2);

                for (n, &xi) in X_row.iter().enumerate() {
                    gradients[n] += error * xi;
                }
            }

            loss /= X.len() as f64;
            for i in 0..self.weights.as_ref().unwrap().len() {
                self.weights.as_mut().unwrap()[i] -=
                    self.learning_rate * gradients[i] / X.len() as f64;
            }
            if verbose == true {
                if i % 100 == 0 {
                    println!("Iteration {}: Loss {}", i, loss);
                }
            }
        }
    }
    fn predict(&self, data: &DataShape) -> Vec<f64> {
        let X = add_bias(&data);
        let weights = self
            .weights
            .as_ref()
            .expect("Train model first before predicting");

        let predictions = X
            .iter()
            .map(|X_row| X_row.iter().zip(weights).map(|(&xi, &wi)| xi * wi).sum())
            .collect();
        return predictions;
    }
}
struct LogisticRegression {
    weights: Option<Vec<f64>>,
    learning_rate: f64,
    iterations: usize,
}

impl LogisticRegression {
    fn new(learning_rate: f64, iterations: usize) -> LogisticRegression {
        LogisticRegression {
            weights: None,
            learning_rate,
            iterations,
        }
    }
    pub fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn fit(&mut self, data: &DataShape, target: &Vec<f64>, verbose: bool) {
        let input_size = match &data {
            DataShape::Vector(vector) => 1,
            DataShape::Matrix(matrix) => matrix[0].len(),
        };

        let X: Vec<Vec<f64>> = add_bias(&data);

        self.weights = Some(vec![0.0; input_size + 1]);
        for i in 0..self.iterations {
            let mut gradients = vec![0.0; self.weights.as_ref().unwrap().len()];
            let mut loss = 0.0;
            for (X_row, &target) in X.iter().zip(target.iter()) {
                let z = X_row
                    .iter()
                    .zip(self.weights.as_ref().unwrap())
                    .map(|(x, y)| x * y)
                    .sum();
                let predicted = Self::sigmoid(z);
                loss += log_loss(predicted, target);

                for (n, &xi) in X_row.iter().enumerate() {
                    gradients[n] += (predicted - target) * xi;
                }
            }

            loss /= X.len() as f64;

            for i in 0..self.weights.as_ref().unwrap().len() {
                self.weights.as_mut().unwrap()[i] -=
                    self.learning_rate * gradients[i] / X.len() as f64;
            }
            if verbose == true {
                if i % 100 == 0 {
                    println!("Iteration {}: Loss {}", i, loss);
                }
            }
        }
    }
    fn predict(&self, data: &DataShape) -> Vec<f64> {
        let X = add_bias(&data);
        X.iter()
            .map(|X_row| {
                Self::sigmoid(
                    X_row
                        .iter()
                        .zip(self.weights.as_ref().unwrap())
                        .map(|(&xi, &wi)| xi * wi)
                        .sum(),
                )
            })
            .collect()
    }
}
fn add_bias(data: &DataShape) -> Vec<Vec<f64>> {
    let biased = match data {
        DataShape::Matrix(matrix) => matrix
            .iter()
            .map(|row| {
                let mut new_row = vec![1.0];
                new_row.extend_from_slice(row);
                new_row
            })
            .collect(),
        DataShape::Vector(vector) => vec![{
            let mut new_row = vec![1.0];
            new_row.extend_from_slice(vector);
            new_row
        }],
    };
    return biased;
}

fn sigmoid_vec(data: &DataShape, weight: f64) -> DataShape {
    let e = std::f64::consts::E;
    let sigmoid = match data {
        DataShape::Vector(vector) => DataShape::Vector(
            vector
                .iter()
                .map(|&x| {
                    let z = x * weight;
                    1.0 / (1.0 + f64::exp(-z))
                })
                .collect(),
        ),
        DataShape::Matrix(matrix) => DataShape::Matrix(
            matrix
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&x| {
                            let z = x * weight;
                            1.0 / (1.0 + f64::exp(-z))
                        })
                        .collect()
                })
                .collect(),
        ),
    };
    return sigmoid;
}
fn standardise(vec: &Vec<f64>) -> Vec<f64> {
    let mean = calculate_mean(&vec);
    let std_dev = calculate_std_dev(&vec);
    return vec.iter().map(|x| (x - mean) / std_dev).collect();
}

fn standardise_matrix(vec: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let transposed_vec = transpose(vec);
    let mut standardised_matrix = Vec::new();
    for row in transposed_vec.iter() {
        let standardised_vec = standardise(row);
        standardised_matrix.push(standardised_vec);
    }
    let reshaped_matrix = transpose(&standardised_matrix);
    return reshaped_matrix;
}

fn calculate_mean(vec: &Vec<f64>) -> f64 {
    return vec.iter().sum::<f64>() / vec.len() as f64;
}

fn calculate_std_dev(vec: &Vec<f64>) -> f64 {
    if vec.is_empty() {
        panic!("Vector is empty");
    }
    let mean: f64 = calculate_mean(&vec);
    let variance: f64 =
        (vec.iter().map(|x| (x - mean) * (x - mean))).sum::<f64>() / (vec.len() - 1) as f64;
    let std_dev = variance.sqrt();
    return std_dev;
}

fn log_loss(x: f64, y: f64) -> f64 {
    let epsilon = 1e-7;
    let probpred = x.max(epsilon).min(1.0 - epsilon);
    return -y * probpred.ln() - (1.0 - y) * (1.0 - probpred).ln();
}

fn matrix_vector_multiply(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<f64> {
    if matrix.is_empty() || matrix[0].len() != vector.len() {
        panic!("Invalid dimensions for matrix-vector multiplication.");
    }

    matrix
        .iter()
        .map(|row| row.iter().zip(vector.iter()).map(|(r, v)| r * v).sum())
        .collect()
}
fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![0.0; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }

    transposed
}
fn main() {
    let mut model = LinearRegression::new(0.1, 100000); // Learning rate, iterations
    let x_train = DataShape::Matrix(standardise_matrix(&vec![
        vec![2.0, 3.0],
        vec![1.0, 4.0],
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ]));

    let y_train = vec![0.0, 0.0, 1.0, 1.0]; // Example binary labels

    model.fit(&x_train, &y_train, true);

    let x_test = vec![vec![1.0, 2.0], vec![4.0, 5.0]];
    let predictions = model.predict(&x_train);
    println!("Predictions: {:?}", predictions);
}

/// Trait for that provides description for models and statistical functions
pub trait FerrousLearn {
    // Sigmoid function
    //! Can intake data inside Vec<Vec<f64>> enum as a vector or vector of vectors
    //
}
#[derive(PartialEq)]
enum Verbosity {
    Verbose,
    Silent,
}

enum SVD {
    Full,
    Randomized,
    Auto,
}

enum WeightingFunction {
    Uniform,
    Distance,
}

enum DistanceMetric {
    Euclidean,
    Manhattan,
}

struct PrincipalComponentAnalysis {
    n_components: usize,
    // svd_solver: SVD,
    tol: f64,
    whiten: bool,
}

struct KNearestNeighboursRegressor {
    k: usize,
    weighting_function: WeightingFunction,
    distance_metric: DistanceMetric,
    x_train: Option<Vec<Vec<f64>>>,
    y_train: Option<Vec<f64>>,
}

impl KNearestNeighboursRegressor {
    pub fn new(
        k: usize,
        weighting_function: WeightingFunction,
        distance_metric: DistanceMetric,
    ) -> KNearestNeighboursRegressor {
        KNearestNeighboursRegressor {
            k,
            weighting_function,
            distance_metric,
            x_train: None,
            y_train: None,
        }
    }

    pub fn fit(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<f64>, verbose: Verbosity) {
        self.x_train = Some(x_train.clone());
        self.y_train = Some(y_train.clone());

        if verbose == Verbosity::Verbose {
            println!("Model is lazy, no computation is done until prediction");
        };
    }
    pub fn predict(&self, prediction_matrix: Vec<Vec<f64>>) -> Vec<f64> {
        let x_train = self
            .x_train
            .clone()
            .expect("Train model first before predicting");
        let y_train = self
            .y_train
            .clone()
            .expect("Train model first before predicting");

        let distance_function = match self.distance_metric {
            DistanceMetric::Euclidean => euclidean_distance,
            DistanceMetric::Manhattan => manhatten_distance,
        };

        let weighting_function = match self.weighting_function {
            WeightingFunction::Uniform => uniform_weighting,
            WeightingFunction::Distance => distance_weighting,
        };

        let mut predictions = Vec::new();

        for row in prediction_matrix.iter() {
            let mut distances = Vec::new();
            for (x_train_row, &y_train_row) in x_train.iter().zip(y_train.iter()) {
                let distance = distance_function(row, x_train_row);
                distances.push((distance, y_train_row));
            }
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            distances.truncate(self.k);

            let weights: Vec<f64> = distances
                .iter()
                .map(|(distance, _)| weighting_function(*distance))
                .collect();
            let votes: Vec<f64> = weights
                .iter()
                .zip(distances.iter())
                .map(|(weight, (_, y_train_row))| *weight * *y_train_row)
                .collect();
            let vote_sum = votes.iter().sum::<f64>();
            let total_weight = weights.iter().sum::<f64>();
            let prediction = if total_weight != 0.0 {
                vote_sum / total_weight
            } else {
                0.0
            };
            predictions.push(prediction);
        }

        return predictions;
    }
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
    fn fit(&mut self, data: &Vec<Vec<f64>>, target: &Vec<f64>, verbose: bool) {
        let input_size = data[0].len();
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
    fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<f64> {
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

    fn fit(&mut self, data: &Vec<Vec<f64>>, target: &Vec<f64>, verbose: bool) {
        let input_size = data[0].len();
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
    fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<f64> {
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
fn add_bias(data: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let biased = data
        .iter()
        .map(|row| {
            let mut new_row = vec![1.0];
            new_row.extend_from_slice(row);
            new_row
        })
        .collect();

    return biased;
}

fn sigmoid_vec(data: &Vec<Vec<f64>>, weight: f64) -> Vec<Vec<f64>> {
    let e = std::f64::consts::E;
    let sigmoid = data
        .iter()
        .map(|row| {
            row.iter()
                .map(|&x| {
                    let z = x * weight;
                    1.0 / (1.0 + f64::exp(-z))
                })
                .collect()
        })
        .collect();
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

fn euclidean_distance(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    if vec1.len() != vec2.len() {
        panic!("Vectors must be of same length");
    }
    if vec1.is_empty() || vec2.is_empty() {
        panic!("Vectors cannot be empty");
    }

    vec1.iter()
        .zip(vec2.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn manhatten_distance(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    if vec1.len() != vec2.len() {
        panic!("Vectors must be of same length");
    }
    if vec1.is_empty() || vec2.is_empty() {
        panic!("Vectors cannot be empty");
    }

    vec1.iter()
        .zip(vec2.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>()
}

fn uniform_weighting(distance: f64) -> f64 {
    1.0
}

fn distance_weighting(distance: f64) -> f64 {
    1.0 / distance
}

fn main() {
    let mut model = LinearRegression::new(0.1, 100000); // Learning rate, iterations
    let x_train = standardise_matrix(&vec![
        vec![2.0, 3.0],
        vec![1.0, 4.0],
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ]);

    let y_train = vec![0.0, 0.0, 1.0, 1.0]; // Example binary labels

    model.fit(&x_train, &y_train, true);

    let x_test = vec![vec![1.0, 2.0], vec![4.0, 5.0]];
    let predictions = model.predict(&x_train);
    println!("Predictions: {:?}", predictions);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_regressor() {
        let mut regressor = KNearestNeighboursRegressor::new(
            3, // Use 3 nearest neighbors
            WeightingFunction::Uniform,
            DistanceMetric::Euclidean,
        );

        // Dummy training data - usually you'd have more complex data
        let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let y_train = vec![2.0, 3.0, 4.0]; // Corresponding labels

        regressor.fit(&x_train, &y_train, Verbosity::Silent);

        // Test prediction
        let predictions = regressor.predict(vec![vec![2.0, 3.0]]); // Predict for a data point similar to training data

        // Check the prediction - in this simple case, it might be close to the average of y_train
        assert_eq!(predictions.len(), 1); // Ensure we have one prediction
        assert!((predictions[0] - 3.0).abs() < 1e-5); // Check if the prediction is as expected
    }
    #[test]
    fn test_linear_regression() {
        // Create dummy data
        let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let target = vec![3.0, 5.0, 7.0]; // Simple linear relationship

        // Create and fit the model
        let mut model = LinearRegression::new(0.01, 1000);
        model.fit(&data, &target, false);

        // Predict using the same data
        let predictions = model.predict(&data);

        // Check predictions (simple check due to randomness in the training process)
        for (predicted, &actual) in predictions.iter().zip(target.iter()) {
            assert!((predicted - actual).abs() < 1.0); // Adjust tolerance as needed
        }
    }
    #[test]
    fn test_logistic_regression() {
        // Create dummy data
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        let target = vec![0.0, 0.0, 1.0]; // Simple binary targets

        // Create and fit the model
        let mut model = LogisticRegression::new(0.01, 1000);
        model.fit(&data, &target, false);

        // Predict using the same data
        let predictions = model.predict(&data);

        // Check predictions (simple check due to randomness in the training process)
        for (predicted, &actual) in predictions.iter().zip(target.iter()) {
            let predicted_class = if *predicted > 0.5 { 1.0 } else { 0.0 };
            assert_eq!(predicted_class, actual);
        }
    }
    #[test]
    fn test_add_bias() {
        let data = vec![vec![2.0, 3.0], vec![4.0, 5.0]];
        let expected = vec![vec![1.0, 2.0, 3.0], vec![1.0, 4.0, 5.0]];
        assert_eq!(add_bias(&data), expected);
    }
    #[test]
    fn test_sigmoid_vec() {
        let data = vec![vec![0.0], vec![1.0]];
        let weight = 1.0;
        let sigmoided = sigmoid_vec(&data, weight);
        let expected = vec![vec![0.5], vec![LogisticRegression::sigmoid(1.0)]];
        assert_eq!(sigmoided, expected);
    }
    #[test]
    fn test_standardise() {
        let vec = vec![1.0, 2.0, 3.0];
        let standardised = standardise(&vec);
        let mean = calculate_mean(&vec);
        let std_dev = calculate_std_dev(&vec);
        let expected = vec
            .iter()
            .map(|x| (x - mean) / std_dev)
            .collect::<Vec<f64>>();
        assert_eq!(standardised.len(), expected.len());
        for (a, b) in standardised.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6); // Using a small tolerance for floating-point comparison
        }
    }
    #[test]
    fn test_standardise_matrix() {
        unimplemented!("Test standardise_matrix");
    }
    #[test]
    fn test_calculate_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let expected = 2.5;
        assert_eq!(calculate_mean(&data), expected);
    }
    #[test]
    fn test_calculate_std_dev() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let expected = 1.2909944487358056;
        assert_eq!(calculate_std_dev(&data), expected);
    }
    #[test]
    fn test_log_loss() {
        let x = 0.5;
        let y = 1.0;
        let expected = 0.6931471805599453;
        assert_eq!(log_loss(x, y), expected);
    }
    #[test]
    fn test_matrix_vector_multiply() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let vector = vec![1.0, 2.0];
        let expected = vec![5.0, 11.0];
        assert_eq!(matrix_vector_multiply(&matrix, &vector), expected);
    }
    #[test]
    fn test_transpose() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let expected = vec![vec![1.0, 3.0], vec![2.0, 4.0]];
        assert_eq!(transpose(&matrix), expected);
    }
    #[test]
    fn test_euclidean_distance() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let expected = 5.196152422706632;
        assert_eq!(euclidean_distance(&vec1, &vec2), expected);
    }
    #[test]
    fn test_manhatten_distance() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let expected = 9.0;
        assert_eq!(manhatten_distance(&vec1, &vec2), expected);
    }
    #[test]
    fn test_uniform_weighting() {
        let distance = 1.0;
        let expected = 1.0;
        assert_eq!(uniform_weighting(distance), expected);
    }
    #[test]
    fn test_distance_weighting() {
        let distance = 0.1;
        let expected = 10.0;
        assert_eq!(distance_weighting(distance), expected);
    }
    #[test]
    fn test_principal_comonent_analysis() {
        let mut pca = PrincipalComponentAnalysis {
            n_components: 2,
            tol: 0.0,
            whiten: false,
        };

        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        pca.fit(&data);
        let expected = vec![vec![-1.0, 0.0], vec![0.0, 0.0], vec![1.0, 0.0]];
        assert_eq!(pca.components, expected);
    }
    #[test]
    fn test_descision_tree_classifier() {
        let mut tree = DecisionTreeClassifier::new(2, 2);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let target = vec![0.0, 1.0, 0.0];
        tree.fit(&data, &target);
        let expected = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
        assert_eq!(tree.left_child.unwrap().data, expected);
    }

    #[test]
    fn test_decision_tree_regressor() {
        let mut tree = DecisionTreeRegressor::new(2, 2);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let target = vec![0.0, 1.0, 0.0];
        tree.fit(&data, &target);
        let expected = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
        assert_eq!(tree.left_child.unwrap().data, expected);
    }
    #[test]
    fn test_random_forest_classifier() {
        let mut forest = RandomForestClassifier::new(2, 2, 2);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let target = vec![0.0, 1.0, 0.0];
        forest.fit(&data, &target);
        let expected = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
        assert_eq!(forest.trees[0].left_child.unwrap().data, expected);
    }

    #[test]
    fn test_random_forest_regressor() {
        let mut forest = RandomForestRegressor::new(2, 2, 2);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let target = vec![0.0, 1.0, 0.0];
        forest.fit(&data, &target);
        let expected = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
        assert_eq!(forest.trees[0].left_child.unwrap().data, expected);
    }

    #[test]
    fn test_kmeans() {
        let mut kmeans = KMeans::new(2, 2);
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        kmeans.fit(&data);
        let expected = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
        assert_eq!(kmeans.centroids, expected);
    }
}

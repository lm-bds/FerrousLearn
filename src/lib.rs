#![allow(unused_imports, unused_variables, non_snake_case)]

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FerrousError {
    EmptyDataset,
    EmptyRow {
        row: usize,
    },
    EmptyTargets,
    RaggedMatrix {
        row: usize,
        expected: usize,
        actual: usize,
    },
    TargetLengthMismatch {
        samples: usize,
        targets: usize,
    },
    NonFiniteInput {
        row: usize,
        column: usize,
    },
    NonFiniteTarget {
        index: usize,
    },
    TargetOutOfRange {
        index: usize,
    },
    InvalidK {
        k: usize,
        sample_count: usize,
    },
    InvalidClusterCount {
        clusters: usize,
        sample_count: usize,
    },
    InvalidPcaComponentCount {
        components: usize,
        feature_count: usize,
    },
    InsufficientSamples {
        samples: usize,
    },
    ZeroVarianceFeature {
        column: usize,
    },
    PredictionBeforeFit,
    FeatureCountMismatch {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for FerrousError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FerrousError::EmptyDataset => write!(f, "dataset cannot be empty"),
            FerrousError::EmptyRow { row } => write!(f, "row {} cannot be empty", row),
            FerrousError::EmptyTargets => write!(f, "targets cannot be empty"),
            FerrousError::RaggedMatrix {
                row,
                expected,
                actual,
            } => write!(
                f,
                "row {} has {} columns but expected {}",
                row, actual, expected
            ),
            FerrousError::TargetLengthMismatch { samples, targets } => write!(
                f,
                "target length {} does not match sample count {}",
                targets, samples
            ),
            FerrousError::NonFiniteInput { row, column } => {
                write!(f, "input at row {}, column {} must be finite", row, column)
            }
            FerrousError::NonFiniteTarget { index } => {
                write!(f, "target at index {} must be finite", index)
            }
            FerrousError::TargetOutOfRange { index } => {
                write!(f, "target at index {} must be within [0, 1]", index)
            }
            FerrousError::InvalidK { k, sample_count } => {
                write!(f, "k={} is invalid for sample count {}", k, sample_count)
            }
            FerrousError::InvalidClusterCount {
                clusters,
                sample_count,
            } => write!(
                f,
                "cluster count {} is invalid for sample count {}",
                clusters, sample_count
            ),
            FerrousError::InvalidPcaComponentCount {
                components,
                feature_count,
            } => write!(
                f,
                "component count {} is invalid for feature count {}",
                components, feature_count
            ),
            FerrousError::InsufficientSamples { samples } => {
                write!(f, "sample count {} is insufficient for PCA", samples)
            }
            FerrousError::ZeroVarianceFeature { column } => {
                write!(f, "feature at column {} has zero variance", column)
            }
            FerrousError::PredictionBeforeFit => {
                write!(f, "model must be fitted before predicting")
            }
            FerrousError::FeatureCountMismatch { expected, actual } => write!(
                f,
                "feature count {} does not match expected {}",
                actual, expected
            ),
        }
    }
}

impl std::error::Error for FerrousError {}

#[derive(PartialEq)]
pub enum Verbosity {
    Verbose,
    Silent,
}

pub enum SVD {
    Full,
    Randomized,
    Auto,
}

pub enum WeightingFunction {
    Uniform,
    Distance,
}

pub enum DistanceMetric {
    Euclidean,
    Manhattan,
}

pub struct LCG {
    multiplier: u64,
    increment: u64,
    modulus: u64,
    seed: u64,
}

impl LCG {
    // Creates a new LCG with given parameters
    pub fn new(multiplier: u64, increment: u64, modulus: u64, seed: u64) -> Self {
        LCG {
            multiplier,
            increment,
            modulus,
            seed,
        }
    }

    // Generates the next number in the sequence
    fn next(&mut self) -> u64 {
        self.seed = (self.multiplier * self.seed + self.increment) % self.modulus;
        self.seed
    }

    // Generates a random number within a specified range
    fn rand_range(&mut self, min: u64, max: u64) -> u64 {
        min + (self.next() % (max - min + 1))
    }
}

fn validate_matrix(matrix: &[Vec<f64>]) -> Result<usize, FerrousError> {
    if matrix.is_empty() {
        return Err(FerrousError::EmptyDataset);
    }

    let expected = matrix[0].len();
    if expected == 0 {
        return Err(FerrousError::EmptyRow { row: 1 });
    }

    for (row_index, row) in matrix.iter().enumerate() {
        if row.is_empty() {
            return Err(FerrousError::EmptyRow { row: row_index + 1 });
        }

        if row.len() != expected {
            return Err(FerrousError::RaggedMatrix {
                row: row_index + 1,
                expected,
                actual: row.len(),
            });
        }

        for (column_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(FerrousError::NonFiniteInput {
                    row: row_index + 1,
                    column: column_index + 1,
                });
            }
        }
    }

    Ok(expected)
}

fn validate_targets(targets: &[f64], sample_count: usize) -> Result<(), FerrousError> {
    if targets.is_empty() {
        return Err(FerrousError::EmptyTargets);
    }

    if targets.len() != sample_count {
        return Err(FerrousError::TargetLengthMismatch {
            samples: sample_count,
            targets: targets.len(),
        });
    }

    for (index, value) in targets.iter().enumerate() {
        if !value.is_finite() {
            return Err(FerrousError::NonFiniteTarget { index: index + 1 });
        }
    }

    Ok(())
}

fn validate_logistic_targets(targets: &[f64]) -> Result<(), FerrousError> {
    for (index, value) in targets.iter().enumerate() {
        if *value < 0.0 || *value > 1.0 {
            return Err(FerrousError::TargetOutOfRange { index: index + 1 });
        }
    }

    Ok(())
}

fn validate_pca_input(data: &[Vec<f64>]) -> Result<usize, FerrousError> {
    let feature_count = validate_matrix(data)?;

    if data.len() < 2 {
        return Err(FerrousError::InsufficientSamples {
            samples: data.len(),
        });
    }

    for column in 0..feature_count {
        let reference = data[0][column];
        if data.iter().skip(1).all(|row| row[column] == reference) {
            return Err(FerrousError::ZeroVarianceFeature { column: column + 1 });
        }
    }

    Ok(feature_count)
}

fn validate_feature_count(expected: usize, actual: usize) -> Result<(), FerrousError> {
    if expected != actual {
        Err(FerrousError::FeatureCountMismatch { expected, actual })
    } else {
        Ok(())
    }
}

pub struct KMeans {
    n_clusters: usize,
    max_iter: usize,
    tolerance: f64,
    centroids: Option<Vec<Vec<f64>>>,
}

impl KMeans {
    pub fn new(n_clusters: usize, max_iter: usize, tolerance: f64) -> KMeans {
        KMeans {
            n_clusters,
            max_iter,
            tolerance,
            centroids: None,
        }
    }
    pub fn fit(&mut self, data: &[Vec<f64>], seed: u64) -> Result<(), FerrousError> {
        let feature_count = validate_matrix(data)?;
        if self.n_clusters == 0 || self.n_clusters > data.len() {
            return Err(FerrousError::InvalidClusterCount {
                clusters: self.n_clusters,
                sample_count: data.len(),
            });
        }

        let mut centroids = Vec::with_capacity(self.n_clusters);
        let mut rng = LCG::new(1664525, 1013904223, 2u64.pow(32), seed);
        for _ in 0..self.n_clusters {
            let random_index = rng.rand_range(0, data.len() as u64 - 1);
            centroids.push(data[random_index as usize].clone());
        }
        for _ in 0..self.max_iter {
            let cluster_assignments = data
                .iter()
                .map(|row| {
                    let distances = find_distance_point_centroids(row, &centroids);

                    find_closest_centroid(&distances)
                })
                .collect();
            let clusters = create_3d_clusters(data, cluster_assignments, self.n_clusters);
            let new_centroids = calculate_new_centroid(&clusters, &centroids);
            let mut centroid_movement = 0.0;
            for (i, centroid) in centroids.iter().enumerate() {
                centroid_movement += vector_difference_norm(centroid, &new_centroids[i]);
            }
            if centroid_movement < self.tolerance {
                break;
            }
            centroids = new_centroids;
        }
        debug_assert!(centroids
            .iter()
            .all(|centroid| centroid.len() == feature_count));
        self.centroids = Some(centroids.clone());
        Ok(())
    }

    pub fn predict(&self, data: &[Vec<f64>]) -> Result<Vec<usize>, FerrousError> {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or(FerrousError::PredictionBeforeFit)?;
        let feature_count = validate_matrix(data)?;
        validate_feature_count(centroids[0].len(), feature_count)?;
        let mut predictions = Vec::new();
        for row in data.iter() {
            let distances = find_distance_point_centroids(row, centroids);
            let closest_centroid = find_closest_centroid(&distances);
            predictions.push(closest_centroid);
        }
        Ok(predictions)
    }
}

pub struct PrincipalComponentAnalysis {
    n_components: usize,
    tolerance: f64,
}
impl PrincipalComponentAnalysis {
    pub fn new(n_components: usize, tolerance: f64) -> PrincipalComponentAnalysis {
        PrincipalComponentAnalysis {
            n_components,
            tolerance,
        }
    }
    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FerrousError> {
        let n_features = validate_pca_input(data)?;
        if self.n_components == 0 || self.n_components > n_features {
            return Err(FerrousError::InvalidPcaComponentCount {
                components: self.n_components,
                feature_count: n_features,
            });
        }

        let data = standardise_matrix(data);
        let covariance_matrix = covariance_matrix(&data);
        let eigenvalues = qr_algorithm(&covariance_matrix, self.tolerance);
        let mut eigenvectors = Vec::new();
        for eigenvalue in eigenvalues.iter() {
            let eigenvector = find_eigenvector(&covariance_matrix, eigenvalue);
            eigenvectors.push(eigenvector);
        }

        let projection_matrix = form_projection_matrix(&eigenvectors, self.n_components);

        Ok(transform_data(&data, &projection_matrix))
    }
}

pub struct KNearestNeighboursRegressor {
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

    pub fn fit(
        &mut self,
        x_train: &[Vec<f64>],
        y_train: &[f64],
        verbose: Verbosity,
    ) -> Result<(), FerrousError> {
        let feature_count = validate_matrix(x_train)?;
        validate_targets(y_train, x_train.len())?;
        if self.k == 0 || self.k > x_train.len() {
            return Err(FerrousError::InvalidK {
                k: self.k,
                sample_count: x_train.len(),
            });
        }

        debug_assert!(feature_count > 0);
        self.x_train = Some(x_train.to_vec());
        self.y_train = Some(y_train.to_vec());

        if verbose == Verbosity::Verbose {
            println!("Model is lazy, no computation is done until prediction");
        };

        Ok(())
    }
    pub fn predict(&self, prediction_matrix: &[Vec<f64>]) -> Result<Vec<f64>, FerrousError> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or(FerrousError::PredictionBeforeFit)?;
        let y_train = self
            .y_train
            .as_ref()
            .ok_or(FerrousError::PredictionBeforeFit)?;

        let distance_function = match self.distance_metric {
            DistanceMetric::Euclidean => euclidean_distance,
            DistanceMetric::Manhattan => manhatten_distance,
        };

        let (weighting_function, prefer_exact_matches) = match self.weighting_function {
            WeightingFunction::Uniform => (uniform_weighting as fn(f64) -> f64, false),
            WeightingFunction::Distance => (distance_weighting as fn(f64) -> f64, true),
        };

        let prediction_feature_count = validate_matrix(prediction_matrix)?;
        validate_feature_count(x_train[0].len(), prediction_feature_count)?;

        let mut predictions = Vec::new();

        for row in prediction_matrix.iter() {
            let mut distances = Vec::new();
            for (x_train_row, &y_train_row) in x_train.iter().zip(y_train.iter()) {
                let distance = distance_function(row, x_train_row);
                distances.push((distance, y_train_row));
            }
            distances.sort_by(|a, b| a.0.total_cmp(&b.0));
            distances.truncate(self.k);

            let zero_distance_targets: Vec<f64> = distances
                .iter()
                .filter(|(distance, _)| *distance == 0.0)
                .map(|(_, target)| *target)
                .collect();

            let prediction = if prefer_exact_matches && !zero_distance_targets.is_empty() {
                zero_distance_targets.iter().sum::<f64>() / zero_distance_targets.len() as f64
            } else {
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
                if total_weight != 0.0 {
                    vote_sum / total_weight
                } else {
                    0.0
                }
            };
            predictions.push(prediction);
        }

        Ok(predictions)
    }
}

pub struct LinearRegression {
    weights: Option<Vec<f64>>,
    learning_rate: f64,
    iterations: usize,
}

impl LinearRegression {
    pub fn new(learning_rate: f64, iterations: usize) -> LinearRegression {
        LinearRegression {
            weights: None,
            learning_rate,
            iterations,
        }
    }
    pub fn fit(
        &mut self,
        data: &[Vec<f64>],
        target: &[f64],
        verbose: bool,
    ) -> Result<(), FerrousError> {
        let input_size = validate_matrix(data)?;
        validate_targets(target, data.len())?;

        let x = add_bias(data);
        let mut weights = vec![0.0; input_size + 1];

        for i in 0..self.iterations {
            let mut gradients = vec![0.0; weights.len()];
            let mut loss = 0.0;
            for (x_row, &target_value) in x.iter().zip(target.iter()) {
                let predicted: f64 = x_row.iter().zip(weights.iter()).map(|(x, y)| x * y).sum();
                let error: f64 = predicted - target_value;
                loss += error.powi(2);

                for (n, &xi) in x_row.iter().enumerate() {
                    gradients[n] += error * xi;
                }
            }

            loss /= x.len() as f64;
            for (weight, gradient) in weights.iter_mut().zip(gradients.iter()) {
                *weight -= self.learning_rate * gradient / x.len() as f64;
            }
            if verbose && i % 100 == 0 {
                println!("Iteration {}: Loss {}", i, loss);
            }
        }

        self.weights = Some(weights);
        Ok(())
    }
    pub fn predict(&self, data: &[Vec<f64>]) -> Result<Vec<f64>, FerrousError> {
        let weights = self
            .weights
            .as_ref()
            .ok_or(FerrousError::PredictionBeforeFit)?;
        let input_size = validate_matrix(data)?;
        validate_feature_count(weights.len() - 1, input_size)?;

        let x = add_bias(data);
        let predictions = x
            .iter()
            .map(|x_row| x_row.iter().zip(weights).map(|(&xi, &wi)| xi * wi).sum())
            .collect();
        Ok(predictions)
    }
}

pub struct LogisticRegression {
    weights: Option<Vec<f64>>,
    learning_rate: f64,
    iterations: usize,
}

impl LogisticRegression {
    pub fn new(learning_rate: f64, iterations: usize) -> LogisticRegression {
        LogisticRegression {
            weights: None,
            learning_rate,
            iterations,
        }
    }
    pub fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    pub fn fit(
        &mut self,
        data: &[Vec<f64>],
        target: &[f64],
        verbose: bool,
    ) -> Result<(), FerrousError> {
        let input_size = validate_matrix(data)?;
        validate_targets(target, data.len())?;
        validate_logistic_targets(target)?;

        let x = add_bias(data);
        let mut weights = vec![0.0; input_size + 1];
        for i in 0..self.iterations {
            let mut gradients = vec![0.0; weights.len()];
            let mut loss = 0.0;
            for (x_row, &target_value) in x.iter().zip(target.iter()) {
                let z = x_row.iter().zip(weights.iter()).map(|(x, y)| x * y).sum();
                let predicted = Self::sigmoid(z);
                loss += log_loss(predicted, target_value);

                for (n, &xi) in x_row.iter().enumerate() {
                    gradients[n] += (predicted - target_value) * xi;
                }
            }

            loss /= x.len() as f64;

            for (weight, gradient) in weights.iter_mut().zip(gradients.iter()) {
                *weight -= self.learning_rate * gradient / x.len() as f64;
            }
            if verbose && i % 100 == 0 {
                println!("Iteration {}: Loss {}", i, loss);
            }
        }

        self.weights = Some(weights);
        Ok(())
    }
    pub fn predict(&self, data: &[Vec<f64>]) -> Result<Vec<f64>, FerrousError> {
        let weights = self
            .weights
            .as_ref()
            .ok_or(FerrousError::PredictionBeforeFit)?;
        let input_size = validate_matrix(data)?;
        validate_feature_count(weights.len() - 1, input_size)?;

        let x = add_bias(data);
        let predictions = x
            .iter()
            .map(|x_row| {
                Self::sigmoid(
                    x_row
                        .iter()
                        .zip(weights.iter())
                        .map(|(&xi, &wi)| xi * wi)
                        .sum(),
                )
            })
            .collect();
        Ok(predictions)
    }
}

fn add_bias(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let biased = data
        .iter()
        .map(|row| {
            let mut new_row = vec![1.0];
            new_row.extend_from_slice(row);
            new_row
        })
        .collect();

    biased
}

fn standardise(vec: &[f64]) -> Vec<f64> {
    let mean = calculate_mean(vec);
    let std_dev = calculate_std_dev(vec);
    vec.iter().map(|x| (x - mean) / std_dev).collect()
}

fn standardise_matrix(vec: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let transposed_vec = transpose(vec);
    let mut standardised_matrix = Vec::new();
    for row in transposed_vec.iter() {
        let standardised_vec = standardise(row);
        standardised_matrix.push(standardised_vec);
    }

    transpose(&standardised_matrix)
}

fn calculate_mean(vec: &[f64]) -> f64 {
    vec.iter().sum::<f64>() / vec.len() as f64
}

fn calculate_std_dev(vec: &[f64]) -> f64 {
    if vec.is_empty() {
        panic!("Vector is empty");
    }
    let mean: f64 = calculate_mean(vec);
    let variance: f64 =
        (vec.iter().map(|x| (x - mean) * (x - mean))).sum::<f64>() / (vec.len() - 1) as f64;

    variance.sqrt()
}

fn log_loss(x: f64, y: f64) -> f64 {
    let epsilon = 1e-7;
    let probpred = x.max(epsilon).min(1.0 - epsilon);
    -y * probpred.ln() - (1.0 - y) * (1.0 - probpred).ln()
}

fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

fn euclidean_distance(vec1: &[f64], vec2: &[f64]) -> f64 {
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

fn manhatten_distance(vec1: &[f64], vec2: &[f64]) -> f64 {
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

fn sdot(vec1: &[f64], vec2: &[f64]) -> f64 {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(x, y)| x * y)
        .sum::<f64>()
}

fn covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_features = data[0].len();
    let n_samples = data.len();
    let mut covariance_matrix = vec![vec![0.0; n_features]; n_features];
    let transposed_data = transpose(data);
    for i in 0..n_features {
        for j in i..n_features {
            let covariance = sdot(&transposed_data[i], &transposed_data[j]) / n_samples as f64;
            covariance_matrix[i][j] = covariance;
            covariance_matrix[j][i] = covariance;
        }
    }
    covariance_matrix
}

fn scale_vector(vec: &[f64], scalar: f64) -> Vec<f64> {
    vec.iter().map(|x| x * scalar).collect()
}

fn vector_difference_norm(vec1: &[f64], vec2: &[f64]) -> f64 {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn gramscmidt_orthogonalisation(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut cols = transpose(matrix);
    let n = cols.len();

    for i in 0..n {
        let n_val = norm(&cols[i]);
        for k in 0..cols[i].len() {
            cols[i][k] /= n_val;
        }
        for j in (i + 1)..n {
            let proj = projection(&cols[j], &cols[i]);
            cols[j] = vector_difference(&cols[j], &proj);
        }
    }

    transpose(&cols)
}

fn calculate_r(matrix: &[Vec<f64>], q: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let q_t = transpose(q);
    let matrix_t = transpose(matrix);
    let n = matrix.len();
    let mut r = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            r[i][j] = sdot(&q_t[i], &matrix_t[j]);
        }
    }

    r
}

fn norm(v: &[f64]) -> f64 {
    sdot(v, v).sqrt()
}
fn vector_difference(vec1: &[f64], vec2: &[f64]) -> Vec<f64> {
    vec1.iter().zip(vec2.iter()).map(|(x, y)| x - y).collect()
}

fn projection(vec1: &[f64], vec2: &[f64]) -> Vec<f64> {
    let scalar = sdot(vec1, vec2) / sdot(vec2, vec2);
    scale_vector(vec2, scalar)
}

fn qr_algorithm(matrix: &[Vec<f64>], tolerance: f64) -> Vec<f64> {
    let mut current_matrix = matrix.to_vec();
    while !has_converged(&current_matrix, tolerance) {
        let q = gramscmidt_orthogonalisation(&current_matrix);
        let r = calculate_r(&current_matrix, &q);
        current_matrix = matrix_multiply(&r, &q);
    }

    (0..current_matrix[0].len())
        .map(|i| current_matrix[i][i])
        .collect()
}

fn has_converged(matrix: &[Vec<f64>], tolerance: f64) -> bool {
    let nrows = matrix.len();
    let ncols = matrix[0].len();

    if nrows != ncols {
        panic!("Matrix must be square.");
    }

    for (i, row) in matrix.iter().enumerate().take(nrows) {
        if row.iter().take(i).any(|value| value.abs() > tolerance) {
            return false;
        }
    }
    true
}

fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; b[0].len()]; a.len()];

    for i in 0..a.len() {
        for j in 0..b[0].len() {
            for k in 0..b.len() {
                let a_part = a[i][k];
                let b_part = b[k][j];
                let result_part = a_part * b_part;
                result[i][j] += result_part;
            }
        }
    }
    result
}

fn gaussian_elimination_for_eigenvector(a: &mut [Vec<f64>]) -> Vec<f64> {
    let n = a.len();
    let mut x = vec![0.0; n];

    // Forward elimination
    for i in 0..n {
        if a[i][i].abs() < 1e-6 {
            a[i][i] = 1.0;
            x[i] = 1.0;
            continue;
        }

        for j in (i + 1)..n {
            let ratio = a[j][i] / a[i][i];
            let (upper, lower) = a.split_at_mut(j);
            let pivot_row = &upper[i];
            let row = &mut lower[0];
            for (k, value) in row.iter_mut().enumerate().skip(i) {
                *value -= ratio * pivot_row[k];
            }
        }
    }

    // Backward substitution
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }

    x
}

fn find_eigenvector(matrix: &[Vec<f64>], eigenvalue: &f64) -> Vec<f64> {
    let mut a = matrix.to_vec();
    let n = a.len();

    // Subtract the eigenvalue from the diagonal elements to form (A - lambda * I)
    for (i, row) in a.iter_mut().enumerate().take(n) {
        row[i] -= *eigenvalue;
    }

    gaussian_elimination_for_eigenvector(&mut a)
}
fn form_projection_matrix(eigenvectors: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    eigenvectors.iter().take(k).cloned().collect()
}

fn transform_data(data: &[Vec<f64>], projection_matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let transposed_projection_matrix = transpose(projection_matrix);
    matrix_multiply(data, &transposed_projection_matrix)
}

fn find_distance_point_centroids(point: &[f64], centroids: &[Vec<f64>]) -> Vec<f64> {
    let distances = centroids
        .iter()
        .map(|centriod| euclidean_distance(point, centriod))
        .collect();
    distances
}

fn find_closest_centroid(distances: &[f64]) -> usize {
    distances
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn create_3d_clusters(
    data: &[Vec<f64>],
    cluster_assignments: Vec<usize>,
    n_cluster: usize,
) -> Vec<Vec<Vec<f64>>> {
    let mut clusters: Vec<Vec<Vec<f64>>> = vec![Vec::new(); n_cluster];

    for (row, &cluster) in data.iter().zip(cluster_assignments.iter()) {
        clusters[cluster].push(row.clone());
    }
    clusters
}

fn calculate_new_centroid(
    clusters: &[Vec<Vec<f64>>],
    previous_centroids: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    clusters
        .iter()
        .enumerate()
        .map(|(index, cluster)| {
            if cluster.is_empty() {
                previous_centroids[index].clone()
            } else {
                average_of_rows(transpose(cluster))
            }
        })
        .collect()
}

fn average_of_rows(matrix: Vec<Vec<f64>>) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            let row_sum: f64 = row.iter().sum();

            row_sum / row.len() as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigmoid_vec(data: &[Vec<f64>], weight: f64) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                row.iter()
                    .map(|&x| {
                        let z = x * weight;
                        1.0 / (1.0 + f64::exp(-z))
                    })
                    .collect()
            })
            .collect()
    }

    fn matrix_vector_multiply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
        if matrix.is_empty() || matrix[0].len() != vector.len() {
            panic!("Invalid dimensions for matrix-vector multiplication.");
        }

        matrix
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(r, v)| r * v).sum())
            .collect()
    }

    #[test]
    fn test_knn_regressor() {
        let mut regressor = KNearestNeighboursRegressor::new(
            3,
            WeightingFunction::Uniform,
            DistanceMetric::Euclidean,
        );

        let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let y_train = vec![2.0, 3.0, 4.0];
        regressor
            .fit(&x_train, &y_train, Verbosity::Silent)
            .unwrap();

        let predictions = regressor.predict(&[vec![2.0, 3.0]]).unwrap();
        assert_eq!(predictions.len(), 1);
        assert!((predictions[0] - 3.0).abs() < 1e-5);
    }
    #[test]
    fn test_linear_regression() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let target = vec![3.0, 5.0, 7.0]; // Simple linear relationship

        let mut model = LinearRegression::new(0.01, 1000);
        model.fit(&data, &target, false).unwrap();

        let predictions = model.predict(&data).unwrap();

        for (predicted, &actual) in predictions.iter().zip(target.iter()) {
            assert!((predicted - actual).abs() < 1.0);
        }
    }
    #[test]
    fn test_logistic_regression() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        let target = vec![0.0, 0.0, 1.0]; // Simple binary targets
        let mut model = LogisticRegression::new(0.01, 1000);
        model.fit(&data, &target, false).unwrap();
        let predictions = model.predict(&data).unwrap();
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
            assert!((a - b).abs() < 1e-6);
        }
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
        let expected = std::f64::consts::LN_2;
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
    // #[test]
    // fn test_principal_comonent_analysis() {
    //     let mut pca = PrincipalComponentAnalysis {
    //         n_components: 2,
    //         tol: 0.0,
    //         whiten: false,
    //     };
    //
    //     let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    //     pca.fit(&data);
    //     let expected = vec![vec![-1.0, 0.0], vec![0.0, 0.0], vec![1.0, 0.0]];
    //     assert_eq!(pca.components, expected);
    // }
    // #[test]
    // fn test_descision_tree_classifier() {
    //     let mut tree = DecisionTreeClassifier::new(2, 2);
    //     let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    //     let target = vec![0.0, 1.0, 0.0];
    //     tree.fit(&data, &target);
    //     let expected = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
    //     assert_eq!(tree.left_child.unwrap().data, expected);
    // }
    //
    // #[test]
    // fn test_decision_tree_regressor() {
    //     let mut tree = DecisionTreeRegressor::new(2, 2);
    //     let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    //     let target = vec![0.0, 1.0, 0.0];
    //     tree.fit(&data, &target);
    //     let expected = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
    //     assert_eq!(tree.left_child.unwrap().data, expected);
    // }
    // #[test]
    // fn test_random_forest_classifier() {
    //     let mut forest = RandomForestClassifier::new(2, 2, 2);
    //     let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    //     let target = vec![0.0, 1.0, 0.0];
    //     forest.fit(&data, &target);
    //     let expected = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
    //     assert_eq!(forest.trees[0].left_child.unwrap().data, expected);
    // }
    //
    // #[test]
    // fn test_random_forest_regressor() {
    //     let mut forest = RandomForestRegressor::new(2, 2, 2);
    //     let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    //     let target = vec![0.0, 1.0, 0.0];
    //     forest.fit(&data, &target);
    //     let expected = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
    //     assert_eq!(forest.trees[0].left_child.unwrap().data, expected);
    // }
    //
    // #[test]
    // fn test_kmeans() {
    //     let mut kmeans = KMeans::new(2, 2);
    //     let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    //     kmeans.fit(&data);
    //     let expected = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
    //     assert_eq!(kmeans.centroids, expected);
    // }
    #[test]
    fn test_qr_algorithm() {
        // Define a simple 2x2 symmetric matrix
        let matrix = vec![vec![4.0, 1.0], vec![1.0, 3.0]];

        // Set a tolerance for convergence
        let tolerance = 1e-6;

        // Call the QR algorithm
        let eigenvalues = qr_algorithm(&matrix, tolerance);

        // Known eigenvalues for this matrix are approximately 4.236 and 2.764
        let known_eigenvalues = [4.6180342, 2.381966];
        println!("Eigenvalues: {:?}", eigenvalues);
        // Check if the calculated eigenvalues are close to the known eigenvalues
        assert_eq!(eigenvalues.len(), known_eigenvalues.len());
        for (calc, known) in eigenvalues.iter().zip(known_eigenvalues.iter()) {
            assert!((calc - known).abs() < tolerance);
        }
    }
    fn create_test_data() -> Vec<Vec<f64>> {
        vec![
            vec![2.5, 2.4],
            vec![0.5, 0.7],
            vec![2.2, 2.9],
            vec![1.9, 2.2],
            vec![3.1, 3.0],
            vec![2.3, 2.7],
            vec![2.0, 1.6],
            vec![1.0, 1.1],
            vec![1.5, 1.6],
            vec![1.1, 0.9],
        ]
    }

    #[test]
    fn test_pca_transform() {
        let pca = PrincipalComponentAnalysis::new(2, 0.01);
        let test_data = create_test_data();
        let transformed_data = pca.transform(&test_data).unwrap();

        assert_eq!(transformed_data.len(), 10); // 10 samples
        assert_eq!(transformed_data[0].len(), 2); // 2 principal components

        let var_first_component = transformed_data
            .iter()
            .map(|row| row[0].powi(2))
            .sum::<f64>()
            / 10.0;
        let var_second_component = transformed_data
            .iter()
            .map(|row| row[1].powi(2))
            .sum::<f64>()
            / 10.0;
        assert!(var_first_component >= var_second_component);

        let dot_product = transformed_data
            .iter()
            .map(|row| row[0] * row[1])
            .sum::<f64>();
        assert!(dot_product.abs() < 1e-6);
    }

    #[test]
    fn pca_respects_requested_component_count() {
        let pca = PrincipalComponentAnalysis::new(1, 0.01);
        let test_data = create_test_data();
        let transformed_data = pca.transform(&test_data).unwrap();
        assert!(transformed_data.iter().all(|row| row.len() == 1));
    }
}

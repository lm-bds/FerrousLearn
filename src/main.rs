#![allow(dead_code, unused_imports, unused_variables, non_snake_case)]

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

struct LCG {
    multiplier: u64,
    increment: u64,
    modulus: u64,
    seed: u64,
}

impl LCG {
    // Creates a new LCG with given parameters
    fn new(multiplier: u64, increment: u64, modulus: u64, seed: u64) -> Self {
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

struct KMeans {
    n_clusters: usize,
    max_iter: usize,
    tolerance: f64,
    centroids: Option<Vec<Vec<f64>>>,
}

impl KMeans {
    fn new(n_clusters: usize, max_iter: usize, tolerance: f64) -> KMeans {
        KMeans {
            n_clusters,
            max_iter,
            tolerance,
            centroids: None,
        }
    }
    fn fit(&mut self, data: &Vec<Vec<f64>>, seed: u64) {
        let mut centroids = Vec::new();
        let mut rng = LCG::new(1664525, 1013904223, 2u64.pow(32), seed);
        for _ in 0..self.n_clusters {
            let random_index = rng.rand_range(0, data.len() as u64);
            centroids.push(data[random_index as usize].clone());
        }
        for _ in 0..self.max_iter {
            let cluster_assignments = data
                .iter()
                .map(|row| {
                    let distances = find_distance_point_centroids(row, &centroids);
                    let closest_centroid = find_closest_centroid(&distances);
                    return closest_centroid;
                })
                .collect();
            let clusters = create_3d_clusters(data.clone(), cluster_assignments, self.n_clusters);
            let new_centroids = calculate_new_centroid(&clusters);
            let mut centroid_movement = 0.0;
            for (i, centroid) in centroids.iter().enumerate() {
                centroid_movement += vector_difference_norm(centroid, &new_centroids[i]);
            }
            if centroid_movement < self.tolerance {
                break;
            }
            centroids = new_centroids;
        }
        self.centroids = Some(centroids.clone());
    }

    fn predict(&self, data: &Vec<Vec<f64>>) -> Vec<usize> {
        let mut predictions = Vec::new();
        let centroids = self
            .centroids
            .as_ref()
            .expect("Train model first before predicting");
        for row in data.iter() {
            let distances = find_distance_point_centroids(row, &centroids);
            let closest_centroid = find_closest_centroid(&distances);
            predictions.push(closest_centroid);
        }
        return predictions;
    }
}

struct PrincipalComponentAnalysis {
    n_components: usize,
    // svd_solver: SVD,
    tol: f64,
    whiten: bool,
    tolerance: f64,
}
impl PrincipalComponentAnalysis {
    fn new(
        n_components: usize,
        tol: f64,
        whiten: bool,
        tolerance: f64,
    ) -> PrincipalComponentAnalysis {
        PrincipalComponentAnalysis {
            n_components,
            tol,
            whiten,
            tolerance,
        }
    }
    fn transform(&self, data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let data = standardise_matrix(&data);
        let n_features = data[0].len();
        let n_samples = data.len();
        let n_components = self.n_components;
        let covariance_matrix = covariance_matrix(&data);
        let eigenvalues = qr_algorithm(&covariance_matrix, self.tolerance);
        let mut eigenvectors = Vec::new();
        for eigenvalue in eigenvalues.iter() {
            let eigenvector = find_eigenvector(&covariance_matrix, eigenvalue);
            eigenvectors.push(eigenvector);
        }

        let projection_matrix = form_projection_matrix(&eigenvectors, 2);

        let transformed_data = transform_data(&data, &projection_matrix);
        transformed_data
    }
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

fn sdot(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(x, y)| x * y)
        .sum::<f64>()
}

fn covariance_matrix(data: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n_features = data[0].len();
    let n_samples = data.len();
    let mut covariance_matrix = vec![vec![0.0; n_features]; n_features];
    let transposed_data = transpose(&data);
    for i in 0..n_features {
        for j in i..n_features {
            let covariance = sdot(&transposed_data[i], &transposed_data[j]) / n_samples as f64;
            covariance_matrix[i][j] = covariance;
            covariance_matrix[j][i] = covariance;
        }
    }
    covariance_matrix
}

fn scale_vector(vec: &Vec<f64>, scalar: f64) -> Vec<f64> {
    vec.iter().map(|x| x * scalar).collect()
}

fn normalise_vector(vec: &Vec<f64>) -> Vec<f64> {
    let norm = vec.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.clone()
    }
}

fn vector_difference_norm(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn qr_decomposition(matrix: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_features = matrix[0].len();
    let n_samples = matrix.len();
    let mut q = matrix.clone();
    let mut r = vec![vec![0.0; n_features]; n_features];

    for i in 0..n_features {
        let mut i_th_column = q.iter().map(|row| row[i]).collect::<Vec<f64>>();
        normalise_vector(&mut i_th_column);
        for k in 0..n_samples {
            q[k][i] = i_th_column[k];
        }
        for j in i + 1..n_features {
            let jth_column = q.iter().map(|row| row[j]).collect::<Vec<f64>>();
            r[i][j] = sdot(&i_th_column, &jth_column);

            for k in 0..n_samples {
                q[k][j] -= r[i][j] * i_th_column[k];
            }
        }
    }
    (q, r)
}

fn gramscmidt_orthogonalisation(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut orthogonalised_matrix = Vec::new();
    let transposed_matrix = transpose(&matrix);
    for row in transposed_matrix.iter() {
        let mut orthogonalised_row = row.clone();
        for v in &orthogonalised_matrix {
            let projection = projection(&orthogonalised_row, &v);
            orthogonalised_row = vector_difference(&orthogonalised_row, &projection);
        }
        let norm_row = norm(&mut orthogonalised_row);
        let orthogonalised_row_normalised =
            orthogonalised_row.iter().map(|&x| x / norm_row).collect();
        orthogonalised_matrix.push(orthogonalised_row_normalised);
    }
    transpose(&orthogonalised_matrix)
}

fn calculate_r(matrix: &Vec<Vec<f64>>, q: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let q = transpose(&q);
    let mut r = vec![vec![0.0; matrix.len()]; matrix.len()];

    for i in 0..matrix.len() {
        for j in i..matrix.len() {
            r[i][j] = sdot(&q[i], &matrix[j]);
        }
    }

    r
}

fn norm(v: &Vec<f64>) -> f64 {
    sdot(v, v).sqrt()
}
fn vector_difference(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
    vec1.iter().zip(vec2.iter()).map(|(x, y)| x - y).collect()
}

fn is_zero_vector(vec: &Vec<f64>) -> bool {
    vec.iter().all(|&x| x.abs() < 1e-10)
}

fn projection(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
    let scalar = sdot(vec1, vec2) / sdot(vec2, vec2);
    scale_vector(vec2, scalar)
}
fn qr_algorithm(matrix: &Vec<Vec<f64>>, tolerance: f64) -> Vec<f64> {
    let mut current_matrix = matrix.clone();
    while !has_converged(&current_matrix, tolerance) {
        let q = gramscmidt_orthogonalisation(&current_matrix);
        let r = calculate_r(&current_matrix, &q);
        current_matrix = matrix_multiply(&r, &q);
    }
    let eigenvalues = (0..current_matrix[0].len())
        .map(|i| current_matrix[i][i])
        .collect();
    eigenvalues
}

fn has_converged(matrix: &[Vec<f64>], tolerance: f64) -> bool {
    let nrows = matrix.len();
    let ncols = matrix[0].len();

    // Check if the matrix is square
    if nrows != ncols {
        panic!("Matrix must be square.");
    }

    for i in 0..nrows {
        for j in 0..ncols {
            if i != j && matrix[i][j].abs() > tolerance {
                return false;
            }
        }
    }
    true
}

fn matrix_multiply(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; b[0].len()]; a.len()];

    for i in 0..a.len() {
        for j in 0..b[0].len() {
            for k in 0..b.len() {
                let a_part = a[i][k];
                let b_part = b[k][j];
                let result_part = a_part * b_part;
                result[i][j] = result[i][j] + result_part;
            }
        }
    }
    result
}
fn determinant(matrix: &Vec<Vec<f64>>) -> f64 {
    let nrows = matrix.len();
    let ncols = matrix[0].len();

    assert_eq!(nrows, ncols, "Matrix must be square.");

    match nrows {
        2 => matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        3 => {
            let a = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]);
            let b = matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]);
            let c = matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
            a - b + c
        }
        _ => {
            let (_, u) = lu_decomposition(matrix);
            determinant_from_lu(&u)
        }
    }
}

fn lu_decomposition(matrix: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = matrix.len();
    assert!(
        n > 0 && matrix[0].len() == n,
        "Matrix must be square and non-empty."
    );

    let mut l = vec![vec![0.0; n]; n];
    let mut u = vec![vec![0.0; n]; n];

    for i in 0..n {
        // Upper Triangular
        for k in i..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[i][j] * u[j][k];
            }
            u[i][k] = matrix[i][k] - sum;
        }

        // Lower Triangular
        for k in i..n {
            if i == k {
                l[i][i] = 1.0; // Diagonal as 1
            } else {
                let mut sum = 0.0;
                for j in 0..i {
                    sum += l[k][j] * u[j][i];
                }
                l[k][i] = (matrix[k][i] - sum) / u[i][i];
            }
        }
    }

    (l, u)
}

fn determinant_from_lu(u: &Vec<Vec<f64>>) -> f64 {
    let mut det = 1.0;
    for i in 0..u.len() {
        det *= u[i][i];
    }
    det
}
fn calculate_eigenvalues(matrix: Vec<Vec<f64>>) -> [f64; 2] {
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[1][0];
    let d = matrix[1][1];

    let trace = a + d;
    let determinant = a * d - b * c;

    let middle_term = (trace / 2.0).powi(2) - determinant;
    let sqrt_middle_term = middle_term.sqrt();

    let eigenvalue1 = trace / 2.0 - sqrt_middle_term;
    let eigenvalue2 = trace / 2.0 + sqrt_middle_term;

    [eigenvalue1, eigenvalue2]
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
            for k in i..n {
                a[j][k] -= ratio * a[i][k];
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

fn find_eigenvector(matrix: &Vec<Vec<f64>>, eigenvalue: &f64) -> Vec<f64> {
    let mut a = matrix.to_vec();
    let n = a.len();

    // Subtract the eigenvalue from the diagonal elements to form (A - lambda * I)
    for i in 0..n {
        a[i][i] -= eigenvalue;
    }

    gaussian_elimination_for_eigenvector(&mut a)
}
fn form_projection_matrix(eigenvectors: &Vec<Vec<f64>>, k: usize) -> Vec<Vec<f64>> {
    let mut projection_matrix = Vec::new();
    for i in 0..k {
        projection_matrix.push(eigenvectors[i].clone());
    }
    projection_matrix
}

fn transform_data(data: &Vec<Vec<f64>>, projection_matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let transposed_projection_matrix = transpose(projection_matrix);
    matrix_multiply(data, &transposed_projection_matrix)
}

fn find_distance_point_centroids(point: &Vec<f64>, centroids: &Vec<Vec<f64>>) -> Vec<f64> {
    let distances = centroids
        .iter()
        .map(|centriod| euclidean_distance(point, centriod))
        .collect();
    return distances;
}

fn find_closest_centroid(distances: &Vec<f64>) -> usize {
    let min_distance = distances
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let min_index = distances.iter().position(|x| x == min_distance).unwrap();
    return min_index;
}

fn create_3d_clusters(
    data: Vec<Vec<f64>>,
    cluster_assignments: Vec<usize>,
    n_cluster: usize,
) -> Vec<Vec<Vec<f64>>> {
    let mut clusters: Vec<Vec<Vec<f64>>> = Vec::new();
    let max_cluster = cluster_assignments.iter().max().unwrap();
    for _ in 0..max_cluster + 1 {
        clusters.push(Vec::new());
    }

    for (row, &cluster) in data.iter().zip(cluster_assignments.iter()) {
        clusters[cluster].push(row.clone());
    }
    return clusters;
}

fn calculate_new_centroid(clusters: &Vec<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
    clusters
        .iter()
        .map(|cluster| {
            let new_centroid = average_of_rows(transpose(cluster));
            return new_centroid;
        })
        .collect()
}

fn average_of_rows(matrix: Vec<Vec<f64>>) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            let row_sum: f64 = row.iter().sum();
            let average = row_sum / row.len() as f64;
            return average;
        })
        .collect()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_regressor() {
        let mut regressor = KNearestNeighboursRegressor::new(
            3,
            WeightingFunction::Uniform,
            DistanceMetric::Euclidean,
        );

        let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let y_train = vec![2.0, 3.0, 4.0];
        regressor.fit(&x_train, &y_train, Verbosity::Silent);

        let predictions = regressor.predict(vec![vec![2.0, 3.0]]);
        assert_eq!(predictions.len(), 1);
        assert!((predictions[0] - 3.0).abs() < 1e-5);
    }
    #[test]
    fn test_linear_regression() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let target = vec![3.0, 5.0, 7.0]; // Simple linear relationship

        let mut model = LinearRegression::new(0.01, 1000);
        model.fit(&data, &target, false);

        let predictions = model.predict(&data);

        for (predicted, &actual) in predictions.iter().zip(target.iter()) {
            assert!((predicted - actual).abs() < 1.0);
        }
    }
    #[test]
    fn test_logistic_regression() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        let target = vec![0.0, 0.0, 1.0]; // Simple binary targets
        let mut model = LogisticRegression::new(0.01, 1000);
        model.fit(&data, &target, false);
        let predictions = model.predict(&data);
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
        let known_eigenvalues = vec![4.23606797749979, 2.76393202250021];
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
        let pca = PrincipalComponentAnalysis::new(2, 0.1, false, 0.01);
        let test_data = create_test_data();
        let transformed_data = pca.transform(test_data);

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
}

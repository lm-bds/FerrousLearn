# FerrousLearn

Free of any dependencies, FerrousLearn is a Rust-based machine learning library focusing on providing efficient and reliable implementations of various algorithms. Our goal is to leverage Rust's performance and safety features to deliver a toolset for data scientists and machine learning engineers.
Without any dependancies this simple approach is also a learning tool for felllow data scientist to get more aquianted with the algorithms we use.

## Features
Linear Regression: Implementation of linear regression for predictive modeling.
Logistic Regression: Binary classification using logistic regression.
K-Nearest Neighbors Regressor: A non-parametric method used for regression tasks.
Principal Component Analysis (PCA): Dimensionality reduction technique. // coming soon 
Various Helper Functions: Including distance metrics, standardization, and matrix operations.

## Installation
To use FerrousLearn in your project, add it as a dependency in your Cargo.toml:

toml
Copy code
[dependencies]
ferrouslearn = { git = "https://github.com/lm-bds/ferrouslearn.git" }
Usage
Here's a quick overview of how you can use some of the features of FerrousLearn:

Linear Regression
``
use ferrouslearn::LinearRegression;

let mut model = LinearRegression::new(0.1, 1000);
let x_train = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
let y_train = vec![5.0, 6.0];

model.fit(&x_train, &y_train, false);
let predictions = model.predict(&vec![vec![2.0, 3.0]]);
``

K-Nearest Neighbors Regressor
```rust
use ferrouslearn::{KNearestNeighboursRegressor, DistanceMetric, WeightingFunction};

let mut knn = KNearestNeighboursRegressor::new(3, WeightingFunction::Uniform, DistanceMetric::Euclidean);
knn.fit(&x_train, &y_train, Verbosity::Silent);
let predictions = knn.predict(&vec![vec![2.0, 3.0]]);
```

## Contributing
Contributions to FerrousLearn are welcome! If you have an idea for an improvement or have found a bug, please open an issue or submit a pull request.

## Developing
```rust

// Clone the repository:
git clone https://github.com/your-username/ferrouslearn.git

// Create a new branch:
// Copy code
git checkout -b feature-your-feature
// Make your changes and write tests to ensure functionality.

// Push your branch and create a pull request.

// Running Tests
// To run tests, use the standard Cargo command:

cargo test
```

License
This project is licensed under MIT License.


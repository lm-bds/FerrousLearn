#![allow(
    dead_code,
    reason = "this helper module is compiled separately for each Criterion benchmark target"
)]

use ferrouslearn::{KMeans, LinearRegression, LogisticRegression, PrincipalComponentAnalysis};

pub const KMEANS_SEED: u64 = 7;
pub const KMEANS_CLUSTERS: usize = 4;

pub fn kmeans_data(samples: usize, features: usize) -> Vec<Vec<f64>> {
    let centres = [
        vec![0.0, 0.5, 1.0, 1.5],
        vec![10.0, 10.5, 11.0, 11.5],
        vec![20.0, 20.5, 21.0, 21.5],
        vec![30.0, 30.5, 31.0, 31.5],
    ];
    (0..samples)
        .map(|index| {
            let cluster = index % KMEANS_CLUSTERS;
            let centre = &centres[cluster];
            (0..features)
                .map(|feature| {
                    let phase = (index as f64 * 0.137 + feature as f64 * 0.413).sin();
                    centre[feature] + phase * 0.05 + cluster as f64 * 0.01
                })
                .collect()
        })
        .collect()
}

pub fn kmeans_prediction_data(samples: usize, features: usize) -> Vec<Vec<f64>> {
    (0..samples)
        .map(|index| {
            let cluster = index % KMEANS_CLUSTERS;
            (0..features)
                .map(|feature| cluster as f64 * 10.0 + feature as f64 * 0.25 + index as f64 * 0.001)
                .collect()
        })
        .collect()
}

pub fn pca_data(samples: usize, features: usize) -> Vec<Vec<f64>> {
    (0..samples)
        .map(|row| {
            let x = row as f64;
            (0..features)
                .map(|feature| {
                    let feature = feature as f64 + 1.0;
                    (x * feature * 0.031).sin() + (x * feature * 0.017).cos() + x * feature * 0.002
                })
                .collect()
        })
        .collect()
}

pub fn regression_features(samples: usize, features: usize) -> Vec<Vec<f64>> {
    (0..samples)
        .map(|row| {
            let x = row as f64;
            (0..features)
                .map(|feature| {
                    let feature = feature as f64 + 1.0;
                    (x * feature * 0.011).sin() + x * feature * 0.004 + feature * 0.5
                })
                .collect()
        })
        .collect()
}

pub fn linear_targets(data: &[Vec<f64>]) -> Vec<f64> {
    data.iter()
        .map(|row| 1.25 + row[0] * 2.5 - row[1] * 1.2 + row[2] * 0.75 + row[3] * 0.33)
        .collect()
}

pub fn logistic_targets(data: &[Vec<f64>]) -> Vec<f64> {
    data.iter()
        .map(|row| {
            let score = -0.5 + row[0] * 0.8 - row[1] * 1.1 + row[2] * 0.6 + row[3] * 0.4;
            if score >= 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

pub fn fitted_linear_model(data: &[Vec<f64>], target: &[f64]) -> LinearRegression {
    let mut model = LinearRegression::new(0.05, 400);
    model.fit(data, target, false).unwrap();
    model
}

pub fn fitted_logistic_model(data: &[Vec<f64>], target: &[f64]) -> LogisticRegression {
    let mut model = LogisticRegression::new(0.08, 500);
    model.fit(data, target, false).unwrap();
    model
}

pub fn fitted_kmeans(data: &[Vec<f64>]) -> KMeans {
    let mut model = KMeans::new(KMEANS_CLUSTERS, 100, 1e-8);
    model.fit(data, KMEANS_SEED).unwrap();
    model
}

pub fn fitted_pca(data: &[Vec<f64>], components: usize) -> PrincipalComponentAnalysis {
    let model = PrincipalComponentAnalysis::with_max_iterations(components, 1e-8, 512);
    model.transform(data).unwrap();
    model
}

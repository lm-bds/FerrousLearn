use ferrouslearn::{
    DistanceMetric, KMeans, KNearestNeighboursRegressor, LinearRegression, LogisticRegression,
    PrincipalComponentAnalysis, Verbosity, WeightingFunction,
};

#[test]
fn public_learning_apis_accept_slices() {
    let x_train: &[Vec<f64>] = &[vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    let y_train: &[f64] = &[2.0, 3.0, 4.0];
    let prediction_matrix: &[Vec<f64>] = &[vec![2.0, 3.0]];
    let data: &[Vec<f64>] = &[vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    let labels: &[f64] = &[3.0, 5.0, 7.0];

    let mut knn =
        KNearestNeighboursRegressor::new(3, WeightingFunction::Uniform, DistanceMetric::Euclidean);
    knn.fit(x_train, y_train, Verbosity::Silent);
    let knn_predictions = knn.predict(prediction_matrix);
    assert_eq!(knn_predictions, vec![3.0]);

    let mut linear = LinearRegression::new(0.01, 1_000);
    linear.fit(data, labels, false);
    let linear_predictions = linear.predict(data);
    assert_eq!(linear_predictions.len(), data.len());
    for (predicted, &actual) in linear_predictions.iter().zip(labels.iter()) {
        assert!(predicted.is_finite());
        assert!((predicted - actual).abs() < 0.25);
    }

    let mut logistic = LogisticRegression::new(0.01, 1_000);
    let logistic_targets: &[f64] = &[0.0, 0.0, 1.0];
    logistic.fit(data, logistic_targets, false);
    let logistic_predictions = logistic.predict(data);
    assert_eq!(logistic_predictions.len(), data.len());
    for predicted in &logistic_predictions {
        assert!(predicted.is_finite());
        assert!((0.0..=1.0).contains(predicted));
    }

    let pca_input: &[Vec<f64>] = &[
        vec![2.5, 2.4],
        vec![0.5, 0.7],
        vec![2.2, 2.9],
        vec![1.9, 2.2],
        vec![3.1, 3.0],
    ];
    let pca = PrincipalComponentAnalysis::new(1, 0.01);
    let transformed = pca.transform(pca_input);
    assert_eq!(transformed.len(), pca_input.len());
    assert!(transformed
        .iter()
        .all(|row| row.len() == 1 && row[0].is_finite()));

    let mut kmeans = KMeans::new(1, 10, 0.0001);
    kmeans.fit(data, 42);
    let kmeans_predictions = kmeans.predict(data);
    assert_eq!(kmeans_predictions, vec![0, 0, 0]);
}

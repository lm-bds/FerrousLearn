use ferrouslearn::{
    DistanceMetric, FerrousError, KMeans, KNearestNeighboursRegressor, LinearRegression,
    LogisticRegression, PrincipalComponentAnalysis, Verbosity, WeightingFunction,
};

fn kmeans_fixture() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 1.0],
        vec![1.2, 0.9],
        vec![8.0, 8.0],
        vec![8.1, 7.9],
        vec![4.0, 4.0],
    ]
}

fn kmeans_identical_fixture() -> Vec<Vec<f64>> {
    vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]]
}

fn pca_fixture() -> Vec<Vec<f64>> {
    vec![
        vec![2.5, 2.4],
        vec![0.5, 0.7],
        vec![2.2, 2.9],
        vec![1.9, 2.2],
        vec![3.1, 3.0],
        vec![2.3, 2.7],
    ]
}

fn pca_two_sample_fixture() -> Vec<Vec<f64>> {
    vec![vec![1.0, 2.0], vec![3.0, 4.0]]
}

#[test]
fn kmeans_zero_max_iterations_returns_typed_convergence_failure() {
    let data = kmeans_fixture();
    let mut model = KMeans::new(2, 0, 1e-12);

    let err = model.fit(&data, 7).unwrap_err();
    assert!(matches!(
        err,
        FerrousError::ConvergenceFailure {
            algorithm: "KMeans",
            max_iterations: 0,
        }
    ));

    let predict_err = model.predict(&data).unwrap_err();
    assert!(matches!(predict_err, FerrousError::PredictionBeforeFit));
}

#[test]
fn kmeans_invalid_cluster_count_wins_over_zero_iteration_budget() {
    let data = kmeans_fixture();

    let mut zero_clusters = KMeans::new(0, 0, 1e-12);
    assert!(matches!(
        zero_clusters.fit(&data, 7),
        Err(FerrousError::InvalidClusterCount {
            clusters: 0,
            sample_count: 5,
        })
    ));

    let mut too_many_clusters = KMeans::new(6, 0, 1e-12);
    assert!(matches!(
        too_many_clusters.fit(&data, 7),
        Err(FerrousError::InvalidClusterCount {
            clusters: 6,
            sample_count: 5,
        })
    ));
}

#[test]
fn kmeans_positive_budget_exhaustion_returns_typed_convergence_failure() {
    let data = kmeans_fixture();
    let mut model = KMeans::new(2, 1, 0.0);

    let err = model.fit(&data, 7).unwrap_err();
    assert!(matches!(
        err,
        FerrousError::ConvergenceFailure {
            algorithm: "KMeans",
            max_iterations: 1,
        }
    ));
}

#[test]
fn kmeans_final_allowed_iteration_converges_and_predicts() {
    let data = kmeans_identical_fixture();
    let mut model = KMeans::new(2, 1, 0.0);

    model.fit(&data, 11).unwrap();

    let predictions = model.predict(&data).unwrap();
    assert_eq!(predictions.len(), data.len());
    assert!(predictions.iter().all(|&cluster| cluster < 2));
}

#[test]
fn kmeans_failed_refit_preserves_previous_predictions() {
    let identical_data = kmeans_identical_fixture();
    let nontrivial_data = kmeans_fixture();
    let probe = nontrivial_data.clone();

    let mut model = KMeans::new(2, 1, 0.0);
    model.fit(&identical_data, 11).unwrap();
    let before = model.predict(&probe).unwrap();

    let err = model.fit(&nontrivial_data, 7).unwrap_err();
    assert!(matches!(
        err,
        FerrousError::ConvergenceFailure {
            algorithm: "KMeans",
            max_iterations: 1,
        }
    ));

    let after = model.predict(&probe).unwrap();
    assert_eq!(after, before);
}

#[test]
fn kmeans_empty_clusters_are_recovered_deterministically() {
    let data = kmeans_identical_fixture();

    let mut first = KMeans::new(2, 10, 1e-12);
    first.fit(&data, 11).unwrap();
    let first_predictions = first.predict(&data).unwrap();

    let mut second = KMeans::new(2, 10, 1e-12);
    second.fit(&data, 11).unwrap();
    let second_predictions = second.predict(&data).unwrap();

    assert_eq!(first_predictions, second_predictions);
    assert!(first_predictions.iter().all(|&cluster| cluster < 2));
}

#[test]
fn kmeans_identical_seeds_produce_identical_assignments() {
    let data = kmeans_fixture();

    let mut first = KMeans::new(2, 20, 1e-12);
    let mut second = KMeans::new(2, 20, 1e-12);

    first.fit(&data, 42).unwrap();
    second.fit(&data, 42).unwrap();

    assert_eq!(
        first.predict(&data).unwrap(),
        second.predict(&data).unwrap()
    );
}

#[test]
fn pca_respects_finite_iteration_bound_and_reports_exhaustion() {
    let data = pca_fixture();
    let pca = PrincipalComponentAnalysis::with_max_iterations(1, 0.0, 1);

    let err = pca.transform(&data).unwrap_err();
    assert!(matches!(
        err,
        FerrousError::ConvergenceFailure {
            algorithm: "PCA/QR",
            max_iterations: 1,
        }
    ));
}

#[test]
fn pca_final_allowed_iteration_converges_with_finite_output() {
    let data = pca_two_sample_fixture();
    let pca = PrincipalComponentAnalysis::with_max_iterations(1, 0.0, 1);

    let transformed = pca.transform(&data).unwrap();
    assert!(transformed.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn pca_zero_budget_rejects_the_same_fixture_before_any_iteration() {
    let data = pca_two_sample_fixture();
    let pca = PrincipalComponentAnalysis::with_max_iterations(1, 0.0, 0);

    let err = pca.transform(&data).unwrap_err();
    assert!(matches!(
        err,
        FerrousError::ConvergenceFailure {
            algorithm: "PCA/QR",
            max_iterations: 0,
        }
    ));
}

#[test]
fn representative_successful_algorithm_outputs_are_finite() {
    let data = kmeans_fixture();

    let mut kmeans = KMeans::new(2, 25, 1e-12);
    kmeans.fit(&data, 3).unwrap();
    let clusters = kmeans.predict(&data).unwrap();
    assert_eq!(clusters.len(), data.len());
    assert!(clusters.iter().all(|&cluster| cluster < 2));

    let pca = PrincipalComponentAnalysis::new(2, 1e-12);
    let transformed = pca.transform(&pca_fixture()).unwrap();
    assert!(transformed.iter().flatten().all(|value| value.is_finite()));

    let mut knn =
        KNearestNeighboursRegressor::new(3, WeightingFunction::Uniform, DistanceMetric::Euclidean);
    let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    let y_train = vec![2.0, 3.0, 4.0];
    knn.fit(&x_train, &y_train, Verbosity::Silent).unwrap();
    let knn_predictions = knn.predict(&[vec![2.0, 3.0]]).unwrap();
    assert!(knn_predictions.iter().all(|value| value.is_finite()));

    let linear_data = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    let linear_targets = vec![3.0, 5.0, 7.0];
    let mut linear = LinearRegression::new(0.01, 200);
    linear.fit(&linear_data, &linear_targets, false).unwrap();
    let linear_predictions = linear.predict(&linear_data).unwrap();
    assert!(linear_predictions.iter().all(|value| value.is_finite()));

    let logistic_data = vec![vec![1.0], vec![2.0], vec![3.0]];
    let logistic_targets = vec![0.0, 0.0, 1.0];
    let mut logistic = LogisticRegression::new(0.01, 200);
    logistic
        .fit(&logistic_data, &logistic_targets, false)
        .unwrap();
    let logistic_predictions = logistic.predict(&logistic_data).unwrap();
    assert!(logistic_predictions.iter().all(|value| value.is_finite()));
}

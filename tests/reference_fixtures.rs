use ferrouslearn::{
    pairwise_distance, standardise, DistanceMetric, FerrousError, KMeans,
    KNearestNeighboursRegressor, LinearRegression, LogisticRegression, PrincipalComponentAnalysis,
    Verbosity, WeightingFunction,
};

const PCA_EPS: f64 = 1e-8;

fn column_means(data: &[Vec<f64>]) -> Vec<f64> {
    let cols = data[0].len();
    (0..cols)
        .map(|col| data.iter().map(|row| row[col]).sum::<f64>() / data.len() as f64)
        .collect()
}

fn covariance(a: &[f64], b: &[f64]) -> f64 {
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum::<f64>()
        / (a.len() as f64 - 1.0)
}

#[test]
fn euclidean_and_manhattan_distance_match_the_3_4_5_fixture() {
    // Derived analytically from the classic 3-4-5 right triangle.
    let left = [0.0, 0.0];
    let right = [3.0, 4.0];

    let euclidean = pairwise_distance(&left, &right, DistanceMetric::Euclidean).unwrap();
    let manhattan = pairwise_distance(&left, &right, DistanceMetric::Manhattan).unwrap();

    assert!((euclidean - 5.0).abs() < 1e-12);
    assert!((manhattan - 7.0).abs() < 1e-12);
    assert_eq!(
        euclidean,
        pairwise_distance(&right, &left, DistanceMetric::Euclidean).unwrap()
    );
    assert_eq!(
        manhattan,
        pairwise_distance(&right, &left, DistanceMetric::Manhattan).unwrap()
    );
    assert_eq!(
        0.0,
        pairwise_distance(&left, &left, DistanceMetric::Euclidean).unwrap()
    );
}

#[test]
fn pairwise_distance_rejects_empty_vectors_length_mismatch_and_nonfinite_values() {
    let left = [1.0, 2.0];
    let right = [3.0];
    let empty: [f64; 0] = [];

    assert_eq!(
        pairwise_distance(&empty, &right, DistanceMetric::Euclidean),
        Err(FerrousError::EmptyVector)
    );
    assert_eq!(
        pairwise_distance(&right, &empty, DistanceMetric::Euclidean),
        Err(FerrousError::EmptyVector)
    );
    let mismatch_error = pairwise_distance(&left, &right, DistanceMetric::Euclidean).unwrap_err();
    assert_eq!(
        mismatch_error,
        FerrousError::VectorLengthMismatch { left: 2, right: 1 }
    );
    assert_eq!(
        mismatch_error.to_string(),
        "left vector has length 2 but right vector has length 1"
    );
    assert_eq!(
        pairwise_distance(&right, &left, DistanceMetric::Euclidean),
        Err(FerrousError::VectorLengthMismatch { left: 1, right: 2 })
    );
    assert_eq!(
        pairwise_distance(&[1.0, f64::NAN], &[1.0, 2.0], DistanceMetric::Euclidean),
        Err(FerrousError::NonFiniteVectorInput { index: 2 })
    );
    assert_eq!(
        pairwise_distance(
            &[1.0, 2.0],
            &[1.0, f64::INFINITY],
            DistanceMetric::Euclidean
        ),
        Err(FerrousError::NonFiniteVectorInput { index: 2 })
    );
}

#[test]
fn standardise_matches_the_sample_fixture() {
    // Column-wise sample standardisation of [[1,2],[2,4],[3,6]] produces
    // means of 2 and 4, sample std-devs of 1 and 2, and rows [-1,-1],[0,0],[1,1].
    let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
    let standardised = standardise(&data).unwrap();

    assert_eq!(
        standardised,
        vec![vec![-1.0, -1.0], vec![0.0, 0.0], vec![1.0, 1.0]]
    );
}

#[test]
fn standardise_rejects_invalid_inputs_and_constant_columns() {
    let one_sample = vec![vec![1.0, 2.0]];
    let one_sample_error = standardise(&one_sample).unwrap_err();
    assert_eq!(
        one_sample_error,
        FerrousError::InsufficientSamples { samples: 1 }
    );
    assert_eq!(
        one_sample_error.to_string(),
        "at least two samples are required; received 1"
    );

    let empty: Vec<Vec<f64>> = vec![];
    assert_eq!(standardise(&empty), Err(FerrousError::EmptyDataset));

    let ragged = vec![vec![1.0, 2.0], vec![3.0]];
    assert_eq!(
        standardise(&ragged),
        Err(FerrousError::RaggedMatrix {
            row: 2,
            expected: 2,
            actual: 1,
        })
    );

    let non_finite = vec![vec![1.0, f64::INFINITY], vec![3.0, 4.0]];
    assert_eq!(
        standardise(&non_finite),
        Err(FerrousError::NonFiniteInput { row: 1, column: 2 })
    );

    let constant_first_column = vec![vec![1.0, 2.0], vec![1.0, 4.0], vec![1.0, 6.0]];
    assert_eq!(
        standardise(&constant_first_column),
        Err(FerrousError::ZeroVarianceFeature { column: 1 })
    );

    let constant_second_column = vec![vec![1.0, 2.0], vec![3.0, 2.0], vec![5.0, 2.0]];
    assert_eq!(
        standardise(&constant_second_column),
        Err(FerrousError::ZeroVarianceFeature { column: 2 })
    );
}

#[test]
fn uniform_knn_uses_the_expected_neighbour_mean() {
    // The query at x=1.5 is closest to x=2.0 and x=0.0 for k=2, so the mean
    // of their targets is (20 + 10) / 2 = 15.
    let x_train = vec![vec![0.0], vec![2.0], vec![5.0]];
    let y_train = vec![10.0, 20.0, 30.0];
    let mut model =
        KNearestNeighboursRegressor::new(2, WeightingFunction::Uniform, DistanceMetric::Euclidean);

    model.fit(&x_train, &y_train, Verbosity::Silent).unwrap();
    let prediction = model.predict(&[vec![1.5]]).unwrap();

    assert_eq!(prediction, vec![15.0]);
    assert!(prediction[0] >= *y_train.iter().min_by(|a, b| a.total_cmp(b)).unwrap());
    assert!(prediction[0] <= *y_train.iter().max_by(|a, b| a.total_cmp(b)).unwrap());
}

#[test]
fn pca_produces_two_uncorrelated_components_on_the_fixture() {
    // Cross-checked with NumPy 2.2.4; the PCA projection is sign-invariant, so we
    // assert the absolute coordinates from the trusted reference fixture.
    let data = vec![
        vec![2.0, 1.0],
        vec![4.0, 2.0],
        vec![5.0, 4.0],
        vec![7.0, 6.0],
    ];
    let pca = PrincipalComponentAnalysis::with_max_iterations(2, 1e-12, 256);
    let transformed = pca.transform(&data).unwrap();

    assert_eq!(transformed.len(), data.len());
    assert_eq!(transformed[0].len(), 2);
    assert!(transformed.iter().flatten().all(|value| value.is_finite()));

    let expected = [
        [2.2156832913419713, 0.18623901573433854],
        [0.8039267517097521, 0.3235422902944901],
        [0.5784329433089035, 0.09804848189364146],
        [2.44117709974282, 0.03925479266651011],
    ];
    for (actual_row, expected_row) in transformed.iter().zip(expected.iter()) {
        for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
            assert!((actual.abs() - expected).abs() < PCA_EPS);
        }
    }

    let first = transformed.iter().map(|row| row[0]).collect::<Vec<_>>();
    let second = transformed.iter().map(|row| row[1]).collect::<Vec<_>>();
    assert!(covariance(&first, &second).abs() < PCA_EPS);
    assert!(column_means(&transformed)
        .into_iter()
        .all(|mean| mean.abs() < 1e-8));
}

#[test]
fn linear_regression_tracks_the_line_y_eq_2x_plus_1() {
    let x_train = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y_train = vec![1.0, 3.0, 5.0, 7.0];
    let mut model = LinearRegression::new(0.1, 5_000);

    model.fit(&x_train, &y_train, false).unwrap();
    let predictions = model.predict(&[vec![4.0], vec![5.0]]).unwrap();

    assert!((predictions[0] - 9.0).abs() < 5e-2);
    assert!((predictions[1] - 11.0).abs() < 5e-2);
}

#[test]
fn logistic_regression_predictions_stay_in_bounds_and_order_monotonically() {
    let x_train = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y_train = vec![0.0, 0.0, 1.0, 1.0];
    let mut model = LogisticRegression::new(0.2, 2_000);

    model.fit(&x_train, &y_train, false).unwrap();
    let predictions = model
        .predict(&[vec![-1.0], vec![0.0], vec![1.0], vec![2.0], vec![3.0]])
        .unwrap();

    assert!(predictions.iter().all(|value| (0.0..=1.0).contains(value)));
    assert!(predictions.windows(2).all(|pair| pair[0] <= pair[1] + 1e-8));
}

#[test]
fn kmeans_keeps_the_expected_pairs_co_clustered() {
    let data = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![10.0, 10.0],
        vec![10.0, 11.0],
    ];
    let mut model = KMeans::new(2, 50, 1e-12);

    model.fit(&data, 7).unwrap();
    let labels = model.predict(&data).unwrap();

    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[2], labels[3]);
    assert_ne!(labels[0], labels[2]);
}

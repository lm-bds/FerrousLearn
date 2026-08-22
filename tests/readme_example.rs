use ferrouslearn::{pairwise_distance, standardise, DistanceMetric, LinearRegression};

#[test]
fn readme_example_stays_compile_valid_and_runnable() {
    let x: &[Vec<f64>] = &[vec![1.0], vec![2.0], vec![3.0]];
    let y: &[f64] = &[2.0, 4.0, 6.0];

    let mut model = LinearRegression::new(0.01, 2_000);
    model.fit(x, y, false).unwrap();
    let predictions = model.predict(x).unwrap();

    assert_eq!(predictions.len(), x.len());
    assert!(predictions.iter().all(|value| value.is_finite()));

    let distance = pairwise_distance(&[1.0, 2.0], &[4.0, 6.0], DistanceMetric::Euclidean).unwrap();
    assert!((distance - 5.0).abs() < 1e-12);

    let scaled = standardise(&[vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]]).unwrap();
    assert_eq!(scaled.len(), 3);
    assert!(scaled.iter().flatten().all(|value| value.is_finite()));
}

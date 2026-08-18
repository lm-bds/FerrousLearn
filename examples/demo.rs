use ferrouslearn::{KMeans, LinearRegression, PrincipalComponentAnalysis};

fn main() {
    println!("FerrousLearn — from-scratch ML demo\n");

    // --- KMeans ---
    let mut kmeans = KMeans::new(2, 10, 0.001);
    let data = vec![
        vec![1.0, 1.0],
        vec![1.5, 2.0],
        vec![9.0, 8.0],
        vec![8.0, 9.0],
    ];
    kmeans.fit(&data, 42);
    println!("KMeans labels: {:?}", kmeans.predict(&data));

    // --- Linear regression ---
    let lr_data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let lr_target = vec![2.0, 4.0, 6.0, 8.0];
    let mut lr = LinearRegression::new(0.01, 2000);
    lr.fit(&lr_data, &lr_target, false);
    println!("LinearRegression preds: {:?}", lr.predict(&lr_data));

    // --- PCA ---
    let pca = PrincipalComponentAnalysis::new(2, 0.1, false, 0.01);
    let pca_out = pca.transform(vec![vec![2.5, 2.4], vec![0.5, 0.7], vec![2.2, 2.9]]);
    println!("PCA(1) transform rows: {}", pca_out.len());

    println!("\nRun `cargo test` to execute the full test suite.");
}

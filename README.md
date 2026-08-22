# FerrousLearn

FerrousLearn is a dependency-free, from-scratch machine-learning library in Rust. It is intentionally educational: the code is small enough to read, the APIs are tested, and the runtime has no third-party dependencies, but it is not a drop-in replacement for production numerical libraries.

## Implemented APIs

Public, slice-based APIs currently include:

- `pairwise_distance(&[f64], &[f64], DistanceMetric) -> Result<f64, FerrousError>`
- `standardise(&[Vec<f64>]) -> Result<Vec<Vec<f64>>, FerrousError>`
- `KMeans`
- `PrincipalComponentAnalysis`
- `KNearestNeighboursRegressor`
- `LinearRegression`
- `LogisticRegression`

Public learning and numerical APIs validate documented shape, finiteness, target-domain, and model-state requirements and return typed `FerrousError` values.

## Install

Use it as a Git dependency or copy the repository into your workspace:

```toml
[dependencies]
ferrouslearn = { git = "https://github.com/lm-bds/FerrousLearn.git" }
```

## Example

This example matches the current constructor and prediction signatures and handles `Result` explicitly:

```rust
use ferrouslearn::{
    DistanceMetric, FerrousError, LinearRegression, pairwise_distance, standardise,
};

fn main() -> Result<(), FerrousError> {
    let x: &[Vec<f64>] = &[vec![1.0], vec![2.0], vec![3.0]];
    let y: &[f64] = &[2.0, 4.0, 6.0];

    let mut model = LinearRegression::new(0.01, 2_000);
    model.fit(x, y, false)?;
    let predictions = model.predict(x)?;

    let distance = pairwise_distance(&[1.0, 2.0], &[4.0, 6.0], DistanceMetric::Euclidean)?;
    let scaled = standardise(&[vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]])?;

    println!("predictions: {predictions:?}");
    println!("distance: {distance}");
    println!("scaled rows: {}", scaled.len());
    Ok(())
}
```

For a runnable end-to-end walkthrough, see `cargo run --example demo`.

## Behaviour notes and limitations

- `KMeans::fit(&[Vec<f64>], seed)` is deterministic for a given input matrix, algorithm settings, and seed.
- KMeans cluster labels are arbitrary cluster IDs; they are not semantic class names.
- Every iterative algorithm has a finite iteration budget. KMeans and PCA also validate finite, non-negative tolerances and return typed `ConvergenceFailure` or `InvalidTolerance` errors instead of looping forever.
- PCA applies sample standardisation internally and uses a bounded, from-scratch QR iteration. It rejects constant features and fewer than two samples, and its basic decomposition is less numerically robust than production linear-algebra libraries.
- KNN is scale-sensitive, has linear query cost in the training set size, and should be standardised before use when features are on different scales.
- Linear regression uses gradient descent with a fixed iteration budget. Like most basic gradient methods, it is sensitive to feature scale and learning rate choice.
- Logistic regression produces bounded scores in `[0, 1]`, but the scores are not calibrated probabilities and the implementation is not a production classifier.
- None of these implementations should be treated as clinically validated, safety-critical, or production-ready.

## Testing and references

The repository includes unit tests, integration tests, property tests, and shrunk regression cases for the numerical edge cases that have been fixed so far. When a bug is found, please add a focused regression test rather than relying on manual checks.

For benchmark methodology and fixture design, see [benches/README.md](benches/README.md).

## Local development

Recommended strict developer commands:

```bash
cargo fmt --all --check
cargo check --all-targets --all-features
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo run --example demo
cargo bench --no-run
cargo generate-lockfile
cargo audit
cargo deny check
```

`cargo audit` and `cargo deny check` require a generated `Cargo.lock`; the CI workflow regenerates it before supply-chain gates so the repository can keep `Cargo.lock` ignored.

## Licence

MIT.

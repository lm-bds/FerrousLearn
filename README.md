# FerrousLearn

FerrousLearn is a dependency-free, from-scratch machine-learning library in Rust. It is an educational implementation: the algorithms are readable and tested, but it is not a replacement for production numerical libraries.

## Implemented algorithms

- K-means clustering
- Principal component analysis (PCA)
- K-nearest-neighbours regression
- Linear regression
- Logistic regression
- Distance, standardisation and matrix helpers

## Use

```toml
[dependencies]
ferrouslearn = { git = "https://github.com/lm-bds/FerrousLearn.git" }
```

```rust
use ferrouslearn::LinearRegression;

let x = vec![vec![1.0], vec![2.0], vec![3.0]];
let y = vec![2.0, 4.0, 6.0];
let mut model = LinearRegression::new(0.01, 2_000);
model.fit(&x, &y, false);
let predictions = model.predict(&x);
```

Run the complete example with `cargo run --example demo`.

## Develop

```bash
cargo fmt --check
cargo test --all-targets
cargo clippy --all-targets
```

The test suite covers all advertised algorithms. Contributions should include a regression test for changed numerical behaviour.

## Limitations

The implementations prioritise clarity over numerical performance. Inputs are represented as nested `Vec<f64>` values, and several invalid-input paths currently panic rather than return typed errors.

## Licence

MIT.

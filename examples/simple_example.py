import numpy as np

from rustgression import TlsRegressor


def main():
    """
    Main function to generate test data, fit a TlsRegressor model,
    and display the estimated parameters and predictions.

    This function performs the following steps:
    1. Generates synthetic linear data with added noise.
    2. Instantiates the TlsRegressor model with the generated data.
    3. Retrieves and prints the estimated slope, intercept, and correlation coefficient.
    4. Makes predictions for specified test values and prints the results.

    Returns
    -------
    None
    """
    # Generate test data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    true_slope = 2.0
    true_intercept = 1.0
    y = true_slope * x + true_intercept + np.random.normal(0, 0.5, 100)

    # Instantiate and fit the TlsRegressor
    model = TlsRegressor(x, y)

    # Retrieve and display parameters using new property method API
    print(f"Estimated slope: {model.slope():.4f} (True value: {true_slope})")
    print(
        f"Estimated intercept: {model.intercept():.4f} (True value: {true_intercept})"
    )
    print(f"Correlation coefficient: {model.r_value():.4f}")

    # Execute predictions
    x_test = np.array([0, 5, 10])
    y_pred = model.predict(x_test)
    print("\nPrediction results:")
    for x_val, y_val in zip(x_test, y_pred, strict=True):
        print(f"x = {x_val:.1f} -> y = {y_val:.4f}")


if __name__ == "__main__":
    main()

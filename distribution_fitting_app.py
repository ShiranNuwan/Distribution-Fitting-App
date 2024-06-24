import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm, lognorm, gamma, weibull_min, exponweib, gengamma
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_func

# Streamlit app
st.title('Distribution Fitting App')

# Sidebar options before file upload
Relative_Spectral_Sensitivity = st.sidebar.slider('Relative Spectral Sensitivity', min_value=0.01, max_value=1.00, value=0.70)
num_bins = st.sidebar.slider('Number of bins', min_value=10, max_value=100, value=50)
hist_color = st.sidebar.color_picker('Pick a color for the histogram', '#000000')
maxfev_value_k_dist = st.sidebar.slider('maxfev for fitting K distribution', min_value=800, max_value=10000, value=800)
maxfev_value_gamma_gamma = st.sidebar.slider('maxfev for fitting Gamma-Gamma distribution', min_value=800, max_value=10000, value=1000)

# File upload
uploaded_file = st.file_uploader('Upload a CSV file', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Ensure the Power column is numeric
    data['Power'] = pd.to_numeric(data['Power'], errors='coerce')

    # Count the number of NaN values in the Power column
    num_nan_values = data['Power'].isna().sum()
    st.write(f"Number of NaN values in the Power column: {num_nan_values}")

    # Drop any rows with NaN values in the Power column (if conversion fails)
    data = data.dropna(subset=['Power'])

    if data.empty:
        st.error("The 'Power' column in the uploaded file is empty or contains only non-numeric values.")
    else:
        # Assuming the Power column is named 'Power'
        output_power = data['Power']

        # Normalize the Power data by dividing by Relative spectral sensitivity
        power_data = (output_power / Relative_Spectral_Sensitivity)

        # Normalize the 'Power' column by dividing by its mean
        normalized_power = power_data / power_data.mean()

        # Plot the normalized histogram
        plt.clf()  # Clear the plot before plotting new data
        hist, bins, _ = plt.hist(normalized_power, bins=num_bins, density=True, alpha=0.75, edgecolor='black', color=hist_color)

        # Calculate bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Add labels and title
        plt.xlabel('Normalized received optical power')
        plt.ylabel('Probability Density')
        plt.title('Normalized Histogram with Fitted Distributions')

        # Define a function to calculate R^2
        def calculate_r_squared(fm, fp):
            ss_reg = np.sum((fm - fp) ** 2)
            ss_tot = np.sum((fm - np.mean(fm)) ** 2)
            r_squared = 1 - (ss_reg / ss_tot)
            return r_squared

        # Plotting functions for each distribution
        def plot_normal():
            mu, std = norm.fit(normalized_power)
            p = norm.pdf(bin_centers, mu, std)
            plt.plot(bin_centers, p, 'r--', linewidth=2, label='Normal')
            r_squared_normal = calculate_r_squared(hist, p)
            return r_squared_normal, mu, std**2

        def plot_lognormal():
            shape, loc, scale = lognorm.fit(normalized_power)
            p = lognorm.pdf(bin_centers, shape, loc, scale)
            plt.plot(bin_centers, p, 'b--', linewidth=2, label='Log-normal')
            r_squared_lognormal = calculate_r_squared(hist, p)
            return r_squared_lognormal, np.log(scale), shape**2

        def plot_gamma():
            a, loc, scale = gamma.fit(normalized_power)
            p = gamma.pdf(bin_centers, a, loc, scale)
            plt.plot(bin_centers, p, 'g--', linewidth=2, label='Gamma')
            r_squared_gamma = calculate_r_squared(hist, p)
            return r_squared_gamma, scale, a

        def plot_weibull():
            c, loc, scale = weibull_min.fit(normalized_power)
            p = weibull_min.pdf(bin_centers, c, loc, scale)
            plt.plot(bin_centers, p, 'y--', linewidth=2, label='Weibull')
            r_squared_weibull = calculate_r_squared(hist, p)
            return r_squared_weibull, scale, c

        def plot_exponweib():
            a, c, loc, scale = exponweib.fit(normalized_power, floc=0)
            p = exponweib.pdf(bin_centers, a, c, loc, scale)
            plt.plot(bin_centers, p, 'c--', linewidth=2, label='Exp. Weibull')
            r_squared_exp_weibull = calculate_r_squared(hist, p)
            return r_squared_exp_weibull, a, scale, c

        def plot_gengamma():
            a, c, loc, scale = gengamma.fit(normalized_power)
            p = gengamma.pdf(bin_centers, a, c, loc, scale)
            plt.plot(bin_centers, p, 'm--', linewidth=2, label='Gen. Gamma')
            r_squared_gen_gamma = calculate_r_squared(hist, p)
            return r_squared_gen_gamma, a, c, scale

        def plot_k():
            def k_dist_pdf(x, mu, sigma, nu):
                return 2 * (x ** nu) * np.exp(- (x ** 2 + mu ** 2) / (2 * sigma ** 2)) * (1 / (sigma ** 2)) ** nu

            params, _ = curve_fit(k_dist_pdf, bin_centers, hist, p0=[1, 1, 1], maxfev=maxfev_value_k_dist)
            p = k_dist_pdf(bin_centers, *params)
            plt.plot(bin_centers, p, 'orange', linewidth=2, label='K dist')
            r_squared_k = calculate_r_squared(hist, p)
            return r_squared_k, params[2], params[1]**2

        def plot_gamma_gamma():
            def gamma_gamma_pdf(x, alpha, beta, a, d):
                return (x ** (a - 1)) * ((1 + (x / beta) ** d) ** - (alpha + a)) * (d * (beta ** -a)) / gamma_func(a)

            params, _ = curve_fit(gamma_gamma_pdf, bin_centers, hist, p0=[1.0, 1.0, 1.0, 1.0], maxfev=maxfev_value_gamma_gamma)
            p = gamma_gamma_pdf(bin_centers, *params)
            plt.plot(bin_centers, p, 'brown', linewidth=2, label='Gamma-Gamma')
            r_squared_gamma_gamma = calculate_r_squared(hist, p)
            return r_squared_gamma_gamma, params[0], params[1], params[2]**2

        # Plot all distributions
        results = []
        results.append(('Normal', plot_normal()))
        results.append(('Log-normal', plot_lognormal()))
        results.append(('Gamma', plot_gamma()))
        results.append(('Weibull', plot_weibull()))
        results.append(('Exp. Weibull', plot_exponweib()))
        results.append(('Gen. Gamma', plot_gengamma()))
        results.append(('K dist', plot_k()))
        results.append(('Gamma-Gamma', plot_gamma_gamma()))

        # Add legend
        plt.legend()

        # Show the plot
        st.pyplot(plt)

        # Print R^2 results and parameters
        st.write("Goodness-of-Fit (R^2) Results:")
        for name, (r2, *params) in results:
            st.write(f"{name} Distribution: R^2 = {r2}")
            if name == "Normal":
                st.write(f"(µX = {params[0]}, σ²X = {params[1]}, σ²I = {params[1]})")
            elif name == "Log-normal":
                st.write(f"(µX = {params[0]}, σ²X = {params[1]}, σ²I = {params[1]})")
            elif name == "Gamma":
                st.write(f"(θ = {params[0]}, k = {params[1]}, σ²I = {params[0]**2})")
            elif name == "K dist":
                st.write(f"(α = {params[0]}, σ²I = {params[1]})")
            elif name == "Weibull":
                st.write(f"(β = {params[0]}, η = {params[1]}, σ²I = {params[0]**2})")
            elif name == "Exp. Weibull":
                st.write(f"(α = {params[0]}, β = {params[1]}, η = {params[2]}, σ²I = {params[1]**2})")
            elif name == "Gamma-Gamma":
                st.write(f"(α = {params[0]}, β = {params[1]}, σ²I = {params[2]})")
            elif name == "Gen. Gamma":
                st.write(f"(a = {params[0]}, d = {params[1]}, p = {params[2]}, σ²I = {params[2]**2})")

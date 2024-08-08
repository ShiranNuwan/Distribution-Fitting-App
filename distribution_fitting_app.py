import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm, lognorm, gamma, weibull_min, exponweib, gengamma
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_func

# Streamlit app
st.title('Distribution Fitting App')

# Options for user input before uploading CSV
Relative_Spectral_Sensitivity = st.sidebar.slider('Relative Spectral Sensitivity', min_value=0.01, max_value=1.00, value=0.70)

# Sidebar input for number of bins using slider and number input
st.sidebar.title("Histogram Settings")

num_bins_slider = st.sidebar.slider('Number of bins (slider)', min_value=10, max_value=100, value=50)
num_bins_input = st.sidebar.number_input('Number of bins (input)', min_value=10, max_value=100, value=num_bins_slider)

# Ensure slider and number input are in sync
if num_bins_slider != num_bins_input:
    num_bins_slider = num_bins_input

maxfev_value_k_dist = st.sidebar.slider('maxfev for fitting K distribution', min_value=800, max_value=10000, value=800)
maxfev_value_gamma_gamma = st.sidebar.slider('maxfev for fitting Gamma-Gamma distribution', min_value=800, max_value=10000, value=1000)
hist_color = st.sidebar.color_picker('Pick a color for the histogram', '#00f900')

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
        power_data = output_power / Relative_Spectral_Sensitivity

        # Normalize the 'Power' column by dividing by its mean
        normalized_power = power_data / power_data.mean()

        # Plot the normalized histogram
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

        # Prepare table data
        table_data = []

        # Fit and plot a normal distribution
        mu, std = norm.fit(normalized_power)
        p = norm.pdf(bin_centers, mu, std)
        plt.plot(bin_centers, p, 'r--', linewidth=2, label='Normal')
        r_squared_normal = calculate_r_squared(hist, p)
        table_data.append(["Normal", r_squared_normal, f"µ = {mu}, σ = {std}"])

        # Fit and plot a log-normal distribution
        shape, loc, scale = lognorm.fit(normalized_power)
        p = lognorm.pdf(bin_centers, shape, loc, scale)
        plt.plot(bin_centers, p, 'b--', linewidth=2, label='Log-normal')
        r_squared_lognormal = calculate_r_squared(hist, p)
        table_data.append(["Log-normal", r_squared_lognormal, f"shape = {shape}, scale = {scale}, loc = {loc}"])

        # Fit and plot a gamma distribution
        a, loc, scale = gamma.fit(normalized_power)
        p = gamma.pdf(bin_centers, a, loc, scale)
        plt.plot(bin_centers, p, 'g--', linewidth=2, label='Gamma')
        r_squared_gamma = calculate_r_squared(hist, p)
        table_data.append(["Gamma", r_squared_gamma, f"a = {a}, scale = {scale}, loc = {loc}"])

        # Fit and plot a Weibull distribution
        c, loc, scale = weibull_min.fit(normalized_power)
        p = weibull_min.pdf(bin_centers, c, loc, scale)
        plt.plot(bin_centers, p, 'y--', linewidth=2, label='Weibull')
        r_squared_weibull = calculate_r_squared(hist, p)
        table_data.append(["Weibull", r_squared_weibull, f"c = {c}, scale = {scale}, loc = {loc}"])

        # Fit and plot an Exponential Weibull distribution
        a, c, loc, scale = exponweib.fit(normalized_power, floc=0)
        p = exponweib.pdf(bin_centers, a, c, loc, scale)
        plt.plot(bin_centers, p, 'c--', linewidth=2, label='Exp. Weibull')
        r_squared_exp_weibull = calculate_r_squared(hist, p)
        table_data.append(["Exp. Weibull", r_squared_exp_weibull, f"a = {a}, c = {c}, scale = {scale}, loc = {loc}"])

        # Fit and plot a Generalized Gamma distribution
        a, c, loc, scale = gengamma.fit(normalized_power)
        p = gengamma.pdf(bin_centers, a, c, loc, scale)
        plt.plot(bin_centers, p, 'm--', linewidth=2, label='Gen. Gamma')
        r_squared_gen_gamma = calculate_r_squared(hist, p)
        table_data.append(["Gen. Gamma", r_squared_gen_gamma, f"a = {a}, c = {c}, scale = {scale}, loc = {loc}"])

        # K distribution fitting
        def k_dist_pdf(x, mu, sigma, nu):
            return 2 * (x ** nu) * np.exp(- (x ** 2 + mu ** 2) / (2 * sigma ** 2)) * (1 / (sigma ** 2)) ** nu

        params, _ = curve_fit(k_dist_pdf, bin_centers, hist, p0=[1, 1, 1], maxfev=maxfev_value_k_dist)
        p = k_dist_pdf(bin_centers, *params)
        plt.plot(bin_centers, p, 'orange', linewidth=2, label='K dist')
        r_squared_k = calculate_r_squared(hist, p)
        table_data.append(["K dist.", r_squared_k, f"mu = {params[0]}, sigma = {params[1]}, nu = {params[2]}"])

        # Gamma-Gamma distribution fitting
        def gamma_gamma_pdf(x, alpha, beta, a, d):
            return (x ** (a - 1)) * ((1 + (x / beta) ** d) ** - (alpha + a)) * (d * (beta ** -a)) / gamma_func(a)

        params, _ = curve_fit(gamma_gamma_pdf, bin_centers, hist, p0=[1.0, 1.0, 1.0, 1.0], maxfev=maxfev_value_gamma_gamma)
        p = gamma_gamma_pdf(bin_centers, *params)
        plt.plot(bin_centers, p, 'brown', linewidth=2, label='Gamma-Gamma')
        r_squared_gamma_gamma = calculate_r_squared(hist, p)
        table_data.append(["Gamma-Gamma", r_squared_gamma_gamma, f"alpha = {params[0]}, beta = {params[1]}, a = {params[2]}, d = {params[3]}"])

        # Add legend
        plt.legend()

        # Show the plot
        st.pyplot(plt)

        # Create the results DataFrame
        columns = ["Distribution", "Goodness-of-Fit (R^2)", "Parameters"]
        results_df = pd.DataFrame(table_data, columns=columns)

        # Display the results DataFrame
        st.subheader("Goodness-of-Fit (R^2) Results:")
        st.dataframe(results_df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm, lognorm, gamma, weibull_min, exponweib, gengamma
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_func

# Streamlit app
st.title('Distribution Fitting App')

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

        # Option to change Relative Spectral Sensitivity
        Relative_Spectral_Sensitivity = st.sidebar.slider('Relative Spectral Sensitivity', min_value=0.01, max_value=1.00, value=0.70)

        # Normalize the Power data by dividing by Relative spectral sensitivity
        power_data = (output_power / Relative_Spectral_Sensitivity)

        # Normalize the 'Power' column by dividing by its mean
        normalized_power = power_data / power_data.mean()

        # Option to change the number of bins
        num_bins = st.sidebar.slider('Number of bins', min_value=10, max_value=100, value=50)

        # Plot the normalized histogram
        hist, bins, _ = plt.hist(normalized_power, bins=num_bins, density=True, alpha=0.75, edgecolor='black')

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

        # Fit and plot a normal distribution
        mu, std = norm.fit(normalized_power)
        p = norm.pdf(bin_centers, mu, std)
        plt.plot(bin_centers, p, 'r--', linewidth=2, label='Normal')
        r_squared_normal = calculate_r_squared(hist, p)
        mu_X_normal = mu
        sigma2_X_normal = std**2

        # Fit and plot a log-normal distribution
        shape, loc, scale = lognorm.fit(normalized_power)
        p = lognorm.pdf(bin_centers, shape, loc, scale)
        plt.plot(bin_centers, p, 'b--', linewidth=2, label='Log-normal')
        r_squared_lognormal = calculate_r_squared(hist, p)
        mu_X_lognormal = np.log(scale)
        sigma2_X_lognormal = shape**2

        # Fit and plot a gamma distribution
        a, loc, scale = gamma.fit(normalized_power)
        p = gamma.pdf(bin_centers, a, loc, scale)
        plt.plot(bin_centers, p, 'g--', linewidth=2, label='Gamma')
        r_squared_gamma = calculate_r_squared(hist, p)
        theta_gamma = scale
        k_gamma = a

        # Fit and plot a Weibull distribution
        c, loc, scale = weibull_min.fit(normalized_power)
        p = weibull_min.pdf(bin_centers, c, loc, scale)
        plt.plot(bin_centers, p, 'y--', linewidth=2, label='Weibull')
        r_squared_weibull = calculate_r_squared(hist, p)
        beta_weibull = scale
        eta_weibull = c

        # Fit and plot an Exponential Weibull distribution
        a, c, loc, scale = exponweib.fit(normalized_power, floc=0)
        p = exponweib.pdf(bin_centers, a, c, loc, scale)
        plt.plot(bin_centers, p, 'c--', linewidth=2, label='Exp. Weibull')
        r_squared_exp_weibull = calculate_r_squared(hist, p)
        alpha_exp_weibull = a
        beta_exp_weibull = scale
        eta_exp_weibull = c

        # Fit and plot a Generalized Gamma distribution
        a, c, loc, scale = gengamma.fit(normalized_power)
        p = gengamma.pdf(bin_centers, a, c, loc, scale)
        plt.plot(bin_centers, p, 'm--', linewidth=2, label='Gen. Gamma')
        r_squared_gen_gamma = calculate_r_squared(hist, p)
        a_gen_gamma = a
        d_gen_gamma = c
        p_gen_gamma = scale

        # Option to change maxfev for K dist.
        maxfev_value_k_dist = st.sidebar.slider('maxfev for fitting K distribution', min_value=800, max_value=10000, value=800)
        
        # Option to change maxfev for Gamma-Gamma
        maxfev_value_gamma_gamma = st.sidebar.slider('maxfev for fitting Gamma-Gamma distribution', min_value=800, max_value=10000,value=1000)
        
        # K distribution fitting
        def k_dist_pdf(x, mu, sigma, nu):
            return 2 * (x ** nu) * np.exp(- (x ** 2 + mu ** 2) / (2 * sigma ** 2)) * (1 / (sigma ** 2)) ** nu

        params, _ = curve_fit(k_dist_pdf, bin_centers, hist, p0=[1, 1, 1], maxfev=maxfev_value_k_dist)
        p = k_dist_pdf(bin_centers, *params)
        plt.plot(bin_centers, p, 'orange', linewidth=2, label='K dist')
        r_squared_k = calculate_r_squared(hist, p)
        alpha_k = params[2]
        sigma2_I_k = params[1]**2

        # Gamma-Gamma distribution fitting
        def gamma_gamma_pdf(x, alpha, beta, a, d):
            return (x ** (a - 1)) * ((1 + (x / beta) ** d) ** - (alpha + a)) * (d * (beta ** -a)) / gamma_func(a)

        params, _ = curve_fit(gamma_gamma_pdf, bin_centers, hist, p0=[1.0, 1.0, 1.0, 1.0], maxfev=maxfev_value_gamma_gamma)
        p = gamma_gamma_pdf(bin_centers, *params)
        plt.plot(bin_centers, p, 'brown', linewidth=2, label='Gamma-Gamma')
        r_squared_gamma_gamma = calculate_r_squared(hist, p)
        alpha_gamma_gamma = params[0]
        beta_gamma_gamma = params[1]
        sigma2_I_gamma_gamma = params[2]**2

        # Add legend
        plt.legend()

        # Show the plot
        st.pyplot(plt)

        # Print R^2 results
        st.write("Goodness-of-Fit (R^2) Results:")
        st.write(f"Normal Distribution: R^2 = {r_squared_normal}")
        st.write(f"Log-normal Distribution: R^2 = {r_squared_lognormal}")
        st.write(f"Gamma Distribution: R^2 = {r_squared_gamma}")
        st.write(f"Weibull Distribution: R^2 = {r_squared_weibull}")
        st.write(f"Exponential Weibull Distribution: R^2 = {r_squared_exp_weibull}")
        st.write(f"Generalized Gamma Distribution: R^2 = {r_squared_gen_gamma}")
        st.write(f"K Distribution: R^2 = {r_squared_k}")
        st.write(f"Gamma-Gamma Distribution: R^2 = {r_squared_gamma_gamma}")

        # Print parameters and GoF
        st.write("\nParameters and Goodness-of-Fit:")
        st.write(f"Normal: GoF = {r_squared_normal}, (µX = {mu_X_normal}, σ²X = {sigma2_X_normal}, σ²I = {sigma2_X_normal})")
        st.write(f"Log-normal: GoF = {r_squared_lognormal}, (µX = {mu_X_lognormal}, σ²X = {sigma2_X_lognormal}, σ²I = {sigma2_X_lognormal})")
        st.write(f"Gamma: GoF = {r_squared_gamma}, (θ = {theta_gamma}, k = {k_gamma}, σ²I = {theta_gamma**2})")
        st.write(f"K dist.: GoF = {r_squared_k}, (α = {alpha_k}, σ²I = {sigma2_I_k})")
        st.write(f"Weibull: GoF = {r_squared_weibull}, (β = {beta_weibull}, η = {eta_weibull}, σ²I = {beta_weibull**2})")
        st.write(f"Exp. Weibull: GoF = {r_squared_exp_weibull}, (α = {alpha_exp_weibull}, β = {beta_exp_weibull}, η = {eta_exp_weibull}, σ²I = {beta_exp_weibull**2})")
        st.write(f"Gamma-Gamma: GoF = {r_squared_gamma_gamma}, (α = {alpha_gamma_gamma}, β = {beta_gamma_gamma}, σ²I = {sigma2_I_gamma_gamma})")
        st.write(f"Generalized Gamma: GoF = {r_squared_gen_gamma}, (a = {a_gen_gamma}, d = {d_gen_gamma}, p = {p_gen_gamma}, σ²I = {p_gen_gamma**2})")

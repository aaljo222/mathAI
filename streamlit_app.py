import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import json

# Navbar with the title "Study Room" and hyperlinks
st.markdown(
    """
    <style>
        .navbar {
            background-color: #f63366;
            padding: 10px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
        }
        .members {
            text-align: center;
            font-size: 18px;
            margin-top: 10px;
        }
        .members a {
            color: black;
            text-decoration: none;
            margin: 0 10px;
        }
        .members a:hover {
            color: #f63366;
            text-decoration: underline;
        }
    </style>
    <div class="navbar">Study Room</div>
    <div class="members">
        <a href="https://example.com/sehyun" target="_blank">Yoo Sehyun</a> | 
        <a href="https://example.com/hyeri" target="_blank">Lee Hyeri</a> | 
        <a href="https://example.com/hyein" target="_blank">Seo Hyein</a>
    </div>
    """,
    unsafe_allow_html=True
)
# Page navigation
page = st.radio("Go to page:", ["1", "2", "3", "4"], horizontal=True)


# Page 1: Vector Addition and Differentiation
if page == "1":
    st.title("Vector Addition and Derivative Visualizer")

    # Vector input
    st.subheader("Enter the coordinates for two vectors:")
    x1, y1 = st.number_input("Vector 1 X component", value=1.0, step=0.1), st.number_input("Vector 1 Y component",
                                                                                           value=2.0, step=0.1)
    x2, y2 = st.number_input("Vector 2 X component", value=3.0, step=0.1), st.number_input("Vector 2 Y component",
                                                                                           value=-0.1, step=0.1)

    # Vector calculations and display
    vector1, vector2 = np.array([x1, y1]), np.array([x2, y2])
    vector_sum = vector1 + vector2
    st.write("Resultant Vector (Sum):", vector_sum)

    # Plot vector addition
    fig, ax = plt.subplots()
    ax.quiver(0, 0, vector1[0], vector1[1], angles='xy', scale_units='xy', scale=1, color="r", label="Vector 1")
    ax.quiver(0, 0, vector2[0], vector2[1], angles='xy', scale_units='xy', scale=1, color="b", label="Vector 2")
    ax.quiver(0, 0, vector_sum[0], vector_sum[1], angles='xy', scale_units='xy', scale=1, color="g", label="Sum")
    max_range = max(np.abs(vector1).max(), np.abs(vector2).max(), np.abs(vector_sum).max()) + 1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect('equal', 'box')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.legend()
    st.pyplot(fig)

    # Differentiation
    st.subheader("Enter a function to differentiate:")
    function_input = st.text_input("Function (in terms of x)", "x**2 + 3*x + 2")
    x = sp.symbols('x')
    try:
        function = sp.sympify(function_input)
        derivative = sp.diff(function, x)
        st.write("Function:", function)
        st.write("Derivative:", derivative)

        # Plot function and derivative
        func_lambda = sp.lambdify(x, function, "numpy")
        derivative_lambda = sp.lambdify(x, derivative, "numpy")
        x_vals = np.linspace(-10, 10, 400)
        y_vals, y_deriv_vals = func_lambda(x_vals), derivative_lambda(x_vals)
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label="Function", color="blue")
        ax.plot(x_vals, y_deriv_vals, label="Derivative", color="orange")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
    except (sp.SympifyError, TypeError):
        st.error("Invalid function input. Please enter a valid mathematical expression.")

# Page 2: Function Integration
elif page == "2":
    st.title("Function Integration Visualizer")

    # Function input for integration
    st.subheader("Enter a function to integrate:")
    function_input = st.text_input("Function (in terms of x)", "x**2 + 3*x + 2")
    lower_limit = st.number_input("Lower limit", value=0.0, step=0.1)
    upper_limit = st.number_input("Upper limit", value=5.0, step=0.1)

    # Integration calculations and plot
    x = sp.symbols('x')
    try:
        function = sp.sympify(function_input)
        integral = sp.integrate(function, x)
        definite_integral = sp.integrate(function, (x, lower_limit, upper_limit))
        st.write("Indefinite Integral:", integral)
        st.write(f"Definite Integral from {lower_limit} to {upper_limit}:", definite_integral)

        # Plot function and integral
        func_lambda = sp.lambdify(x, function, "numpy")
        integral_lambda = sp.lambdify(x, integral, "numpy")
        x_vals = np.linspace(lower_limit, upper_limit, 400)
        y_vals, y_integral_vals = func_lambda(x_vals), integral_lambda(x_vals)
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label="Function", color="blue")
        ax.plot(x_vals, y_integral_vals, label="Indefinite Integral", color="green")
        plt.fill_between(x_vals, y_vals, color="blue", alpha=0.1, label="Area under curve")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
    except (sp.SympifyError, TypeError):
        st.error("Invalid function input. Please enter a valid mathematical expression.")

# Page 3: Eigenvalues and Eigenvectors with Elliptical Data
elif page == "3":
    st.title("Eigenvalues and Eigenvectors of a 2x2 Matrix with Elliptical Data")

    # Matrix input
    st.subheader("Enter the values for a 2x2 matrix:")
    a, b, c, d = st.number_input("Matrix element a", value=1.0), st.number_input("Matrix element b",
                                                                                 value=0.5), st.number_input(
        "Matrix element c", value=0.5), st.number_input("Matrix element d", value=2.0)
    matrix = np.array([[a, b], [c, d]])

    # Generate and transform data points
    np.random.seed(0)
    theta = np.linspace(0, 2 * np.pi, 100)
    x, y = 2 * np.cos(theta) + np.random.normal(scale=0.1, size=theta.shape), np.sin(theta) + np.random.normal(
        scale=0.1, size=theta.shape)
    points = np.vstack([x, y])
    transformed_points = matrix @ points

    # Eigenvalue and eigenvector calculations
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    st.write("Matrix:", matrix)
    st.write("Eigenvalues:", eigenvalues)
    st.write("Eigenvectors:", eigenvectors)

    # Plot transformed data and eigenvectors
    fig, ax = plt.subplots()
    ax.scatter(transformed_points[0, :], transformed_points[1, :], color='skyblue', alpha=0.7,
               label="Transformed Points")
    origin = [0, 0]
    ax.quiver(*origin, *(eigenvectors[:, np.argmax(eigenvalues)] * eigenvalues.max()), angles='xy', scale_units='xy',
              scale=1, color="red", label="Max Eigenvector")
    ax.quiver(*origin, *(eigenvectors[:, np.argmin(eigenvalues)] * eigenvalues.min()), angles='xy', scale_units='xy',
              scale=1, color="green", label="Min Eigenvector")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

# Page 4: Euler's Formula Visualization with Animation
if page == "4":
    st.title("Euler's Formula Data for Animation")

    # Define the function to calculate Euler points on the unit circle
    def calculate_euler_points(steps=100):
        theta_values = np.linspace(0, 2 * np.pi, steps)
        points = [{"theta": theta, "cos": np.cos(theta), "sin": np.sin(theta)} for theta in theta_values]
        return points

    # Provide data to the frontend in JSON format
    if st.button("Generate Euler Points"):
        euler_data = calculate_euler_points()
        st.write("Sample Data Points for Euler's Animation:")
        st.json(euler_data)  # Display JSON format in Streamlit

        # Show the data as downloadable JSON
        st.download_button(
            label="Download Euler Points",
            data=json.dumps(euler_data),
            file_name="euler_points.json",
            mime="application/json"
        )